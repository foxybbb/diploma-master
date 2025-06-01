/* ===========================================================================
** Copyright (C) 2021 Infineon Technologies AG
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
**
** 1. Redistributions of source code must retain the above copyright notice,
**    this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
** 3. Neither the name of the copyright holder nor the names of its
**    contributors may be used to endorse or promote products derived from
**    this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
** LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
** CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
** INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
** CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
** ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
** POSSIBILITY OF SUCH DAMAGE.
** ===========================================================================
*/

/**
 * @file    radar_node.cpp
 *
 * @brief   ROS 2 node that acquires raw data from an Infineon BGT60TR13C radar
 * sensor, processes it with the Range-Doppler Map (RDM) algorithm and
 * 2-D MTI filtering, and publishes the strongest target as
 * sensor_msgs/PointCloud2.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>  // For FFT data
#include <vector>
#include <cmath>
#include <cstring>   // std::memcpy
#include <stdexcept> // std::runtime_error
#include <string>
#include <thread>

// ── Infineon Radar SDK ───────────────────────────────────────────────────────
extern "C" {
#include "ifxAlgo/Algo.h"
#include "ifxAvian/Avian.h"
#include "ifxRadar/Radar.h"
#include "range_doppler_map_defaults.h"
#include "ifxBase/Error.h"
#include "ifxBase/Matrix.h"
#include "ifxBase/Cube.h"
#include "ifxBase/Defines.h" // IFX_OK
}

// ── Error‑handling helpers ───────────────────────────────────────────────────
#ifndef IFX_ERR_BRN_EXIT
#define IFX_ERR_BRN_EXIT()   \
    do {                    \
        if (ifx_error_get())\
            return ifx_error_get();\
    } while (0)
#endif

#define IFX_ERR_CTOR_THROW(msg)                                               \
    if (ifx_error_get())                                                      \
        throw std::runtime_error(std::string(msg) + ": " +                  \
                                 ifx_error_to_string(ifx_error_get()))

// ╔══════════════════════════════════════╗
// ║   Range‑Doppler processing context   ║
// ╚══════════════════════════════════════╝
namespace {
struct RdmContext
{
    ifx_Avian_Device_t* device{};
    ifx_RDM_t*          rdm_handle{};
    ifx_2DMTI_R_t*      mti_handle{};
    ifx_Matrix_R_t*     rdm_mat{};
    ifx_Math_Axis_Spec_t range_spec{};
    ifx_Math_Axis_Spec_t speed_spec{};
};

ifx_Error_t rdm_config(RdmContext& ctx, ifx_Avian_Device_t* dev, ifx_Avian_Config_t* cfg)
{
    if (!cfg) {
        ifx_error_set(IFX_ERROR_ARGUMENT_NULL);
        return IFX_ERROR_ARGUMENT_NULL;
    }

    uint32_t rng_fft = cfg->num_samples_per_chirp * 4;
    uint32_t dop_fft = cfg->num_chirps_per_frame * 4;

    ifx_PPFFT_Config_t rng_fft_cfg = {IFX_FFT_TYPE_R2C, rng_fft, true,
                                      {IFX_WINDOW_BLACKMANHARRIS, cfg->num_samples_per_chirp, 0, true},
                                      true};
    ifx_PPFFT_Config_t dop_fft_cfg = {IFX_FFT_TYPE_C2C, dop_fft, true,
                                      {IFX_WINDOW_CHEBYSHEV, cfg->num_chirps_per_frame, 100, true},
                                      true};

    ifx_RDM_Config_t rdm_cfg = {IFX_SPECT_THRESHOLD, IFX_SCALE_TYPE_LINEAR,
                                rng_fft_cfg, dop_fft_cfg};

    ctx.rdm_handle = ifx_rdm_create(&rdm_cfg);                IFX_ERR_BRN_EXIT();
    ctx.rdm_mat    = ifx_mat_create_r(rng_fft / 2, dop_fft);  IFX_ERR_BRN_EXIT();
    ctx.mti_handle = ifx_2dmti_create_r(IFX_ALPHA_MTI_FILTER,
                                        IFX_MAT_ROWS(ctx.rdm_mat),
                                        IFX_MAT_COLS(ctx.rdm_mat));         IFX_ERR_BRN_EXIT();

    float bw_Hz     = ifx_devconf_get_bandwidth(cfg);                             IFX_ERR_BRN_EXIT();
    ctx.range_spec  = ifx_spectrum_axis_calc_range_axis(IFX_FFT_TYPE_R2C, rng_fft,
                                                        cfg->num_samples_per_chirp, bw_Hz);
    float fc_Hz     = static_cast<float>(ifx_avian_get_sampling_center_frequency(dev, cfg));
    float chirp_s   = ifx_devconf_get_chirp_time(cfg);                            IFX_ERR_BRN_EXIT();
    ctx.speed_spec  = ifx_spectrum_axis_calc_speed_axis(IFX_FFT_TYPE_C2C, dop_fft,
                                                        fc_Hz, chirp_s);
    return ifx_error_get();
}

void rdm_cleanup(RdmContext& ctx)
{
    ifx_rdm_destroy(ctx.rdm_handle);
    ifx_mat_destroy_r(ctx.rdm_mat);
    ifx_2dmti_destroy_r(ctx.mti_handle);
}

void peak_search(const ifx_Matrix_R_t* m, uint32_t& rmx, uint32_t& cmx)
{
    rmx = cmx = 0;
    auto max_v = IFX_MAT_AT(m, 0, 0);
    for (uint32_t r = 0; r < IFX_MAT_ROWS(m); ++r)
        for (uint32_t c = 0; c < IFX_MAT_COLS(m); ++c)
            if (auto v = IFX_MAT_AT(m, r, c); v > max_v) {
                max_v = v; rmx = r; cmx = c;
            }
}

ifx_Error_t rdm_process(RdmContext& ctx, ifx_Cube_R_t* frame,
                        float& rng_m, float& spd_mps)
{
    ifx_Matrix_R_t row;
    ifx_cube_get_row_r(frame, 0, &row);                    IFX_ERR_BRN_EXIT();
    ifx_rdm_run_r(ctx.rdm_handle, &row, ctx.rdm_mat);      IFX_ERR_BRN_EXIT();
    ifx_2dmti_run_r(ctx.mti_handle, ctx.rdm_mat, ctx.rdm_mat); IFX_ERR_BRN_EXIT();

    uint32_t r, c; peak_search(ctx.rdm_mat, r, c);
    rng_m = r * ctx.range_spec.value_bin_per_step;
    spd_mps = ((static_cast<float>(IFX_MAT_COLS(ctx.rdm_mat)) / 2.f) - c) *
              ctx.speed_spec.value_bin_per_step;
    return ifx_error_get();
}
} // namespace

// ╔══════════════════════════════════════╗
// ║            ROS2 Radar Node           ║
// ╚══════════════════════════════════════╝
class RadarPublisherNode : public rclcpp::Node
{
public:
    RadarPublisherNode() : Node("bgt60tr13c_radar_node")
    {
        // ── Open device ────────────────────────────────────────────────────
        ctx_.device = ifx_avian_create();
        if (!ctx_.device) {
            throw std::runtime_error("Failed to create Avian device: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // ── Configure ─────────────────────────────────────────────────────
        ifx_Avian_Config_t cfg{};
        ifx_avian_get_config_defaults(ctx_.device, &cfg);
        if (ifx_error_get() != IFX_OK) {
            throw std::runtime_error("Failed to get default config: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Configure radar parameters
        cfg.sample_rate_Hz = 1000000;        // 1 MHz sampling rate
        cfg.rx_mask = 1;                     // Use RX1 antenna
        cfg.tx_mask = 1;                     // Use TX1 antenna
        cfg.tx_power_level = 31;             // Maximum TX power
        cfg.if_gain_dB = 33;                 // IF gain (18dB HP + 15dB VGA)
        
        // Frame and chirp timing
        cfg.frame_repetition_time_s = 0.2f;  // 5 Hz frame rate
        cfg.chirp_repetition_time_s = 0.001f; // 1ms between chirps
        
        // Anti-aliasing and high-pass filters
        cfg.aaf_cutoff_Hz = 500000;          // 500 kHz anti-aliasing filter
        cfg.hp_cutoff_Hz = 80000;            // 80 kHz high-pass filter
        
        // FMCW parameters
        cfg.start_frequency_Hz = 58000000000; // 58 GHz start frequency
        cfg.end_frequency_Hz = 63000000000;   // 63 GHz end frequency
        
        // Frame structure
        cfg.num_samples_per_chirp = 64;      // 64 samples per chirp
        cfg.num_chirps_per_frame = 32;       // 32 chirps per frame
        
        // MIMO mode
        cfg.mimo_mode = IFX_MIMO_TDM;        // Time-division multiplexing

        // Set the config
        ifx_avian_set_config(ctx_.device, &cfg);
        if (ifx_error_get() != IFX_OK) {
            throw std::runtime_error("Failed to set config: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Verify the actual configuration
        ifx_Avian_Config_t actual_cfg{};
        ifx_avian_get_config(ctx_.device, &actual_cfg);
        if (ifx_error_get() != IFX_OK) {
            throw std::runtime_error("Failed to get actual config: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Log the actual configuration
        RCLCPP_INFO(get_logger(), "Radar configured with:");
        RCLCPP_INFO(get_logger(), "  Frame rate: %.1f Hz", 1.0f/actual_cfg.frame_repetition_time_s);
        RCLCPP_INFO(get_logger(), "  Chirp rate: %.1f Hz", 1.0f/actual_cfg.chirp_repetition_time_s);
        RCLCPP_INFO(get_logger(), "  IF gain: %u dB", actual_cfg.if_gain_dB);
        RCLCPP_INFO(get_logger(), "  HP cutoff: %u Hz", actual_cfg.hp_cutoff_Hz);
        RCLCPP_INFO(get_logger(), "  AAF cutoff: %u Hz", actual_cfg.aaf_cutoff_Hz);

        // Initialize RDM processing
        if (rdm_config(ctx_, ctx_.device, &actual_cfg) != IFX_OK) {
            throw std::runtime_error("Failed to configure RDM: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Create frame buffer
        frame_ = ifx_cube_create_r(actual_cfg.num_samples_per_chirp,
                                  actual_cfg.num_chirps_per_frame, 1);
        if (!frame_) {
            throw std::runtime_error("Failed to create frame buffer: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Start acquisition
        ifx_avian_start_acquisition(ctx_.device);
        if (ifx_error_get() != IFX_OK) {
            throw std::runtime_error("Failed to start acquisition: " + 
                                   std::string(ifx_error_to_string(ifx_error_get())));
        }

        // Create ROS 2 publishers
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("radar/targets", 20);
        pub_fft_ = create_publisher<std_msgs::msg::Float32MultiArray>("radar/fft", 20);
        
        // Set timer to match frame rate
        timer_ = create_wall_timer(
            std::chrono::microseconds(static_cast<int64_t>(actual_cfg.frame_repetition_time_s * 1000000)),
            std::bind(&RadarPublisherNode::timer_cb, this));

        RCLCPP_INFO(get_logger(), "Radar initialized successfully");
    }

    ~RadarPublisherNode() override
    {
        RCLCPP_INFO(get_logger(), "Stopping radar and cleaning up resources...");
        
        if (ctx_.device) {
            ifx_avian_stop_acquisition(ctx_.device);
            if (ifx_error_get() != IFX_OK) {
                RCLCPP_ERROR(get_logger(), "Error stopping acquisition: %s", 
                           ifx_error_to_string(ifx_error_get()));
                ifx_error_clear();
            }
        }

        rdm_cleanup(ctx_);
        ifx_cube_destroy_r(frame_);
        
        if (ctx_.device) {
            ifx_avian_destroy(ctx_.device);
            ctx_.device = nullptr;
        }
        
        RCLCPP_INFO(get_logger(), "Radar resources cleaned up");
    }

private:
    void timer_cb()
    {
        // Get next frame with proper error checking and retry logic
        ifx_Cube_R_t* frame = nullptr;
        int retry_count = 0;
        const int max_retries = 3;

        while (retry_count < max_retries) {
            frame = ifx_avian_get_next_frame(ctx_.device, frame_);
            if (frame) break;
            
            if (ifx_error_get() != IFX_OK) {
                RCLCPP_WARN(get_logger(), "Frame acquisition attempt %d failed: %s", 
                           retry_count + 1, ifx_error_to_string(ifx_error_get()));
                ifx_error_clear();
            }
            retry_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (!frame) {
            RCLCPP_ERROR(get_logger(), "Failed to acquire frame after %d attempts", max_retries);
            return;
        }

        float range_m = 0.0f;
        float speed_m_s = 0.0f;
        
        if (rdm_process(ctx_, frame, range_m, speed_m_s) != IFX_OK) {
            RCLCPP_ERROR(get_logger(), "Error in RDM processing: %s", 
                        ifx_error_to_string(ifx_error_get()));
            ifx_error_clear();
            return;
        }

        publish_point(range_m, speed_m_s);
        publish_fft_data();
    }

    void publish_point(float rng, float spd)
    {
        auto msg = sensor_msgs::msg::PointCloud2();
        msg.header.stamp = now();
        msg.header.frame_id = "radar_link";
        msg.height = msg.width = 1;
        msg.is_dense = true; msg.is_bigendian = false;
        msg.point_step = 4 * sizeof(float);
        msg.row_step   = msg.point_step;
        msg.fields = {
            make_field("x", 0), make_field("y", 4),
            make_field("z", 8), make_field("velocity", 12)};
        std::vector<float> pt = {rng, 0.f, 0.f, spd};
        msg.data.resize(msg.point_step);
        std::memcpy(msg.data.data(), pt.data(), msg.point_step);
        pub_->publish(msg);
    }

    void publish_fft_data()
    {
        auto msg = std_msgs::msg::Float32MultiArray();
        
        // Set layout information
        msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        msg.layout.dim[0].label = "range";
        msg.layout.dim[0].size = IFX_MAT_ROWS(ctx_.rdm_mat);
        msg.layout.dim[0].stride = IFX_MAT_ROWS(ctx_.rdm_mat) * IFX_MAT_COLS(ctx_.rdm_mat);
        msg.layout.dim[1].label = "doppler";
        msg.layout.dim[1].size = IFX_MAT_COLS(ctx_.rdm_mat);
        msg.layout.dim[1].stride = IFX_MAT_COLS(ctx_.rdm_mat);

        // Copy FFT data
        const size_t total_size = IFX_MAT_ROWS(ctx_.rdm_mat) * IFX_MAT_COLS(ctx_.rdm_mat);
        msg.data.resize(total_size);
        
        // Copy matrix data row by row
        for (uint32_t r = 0; r < IFX_MAT_ROWS(ctx_.rdm_mat); ++r) {
            for (uint32_t c = 0; c < IFX_MAT_COLS(ctx_.rdm_mat); ++c) {
                msg.data[r * IFX_MAT_COLS(ctx_.rdm_mat) + c] = IFX_MAT_AT(ctx_.rdm_mat, r, c);
            }
        }

        pub_fft_->publish(msg);
    }

    static sensor_msgs::msg::PointField make_field(const char* name, uint32_t offset)
    {
        sensor_msgs::msg::PointField f;
        f.name = name; f.offset = offset; f.count = 1;
        f.datatype = sensor_msgs::msg::PointField::FLOAT32;
        return f;
    }

    // ── members ───────────────────────────────────────────────────────────
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_fft_;
    rclcpp::TimerBase::SharedPtr timer_;

    RdmContext ctx_{};
    ifx_Cube_R_t* frame_{};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<RadarPublisherNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("radar_main"), "Фатальная ошибка: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
