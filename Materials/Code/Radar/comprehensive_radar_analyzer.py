# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from ifxAvian import Avian
from helpers.fft_spectrum import *
from helpers.DBF import DBF
from helpers.DopplerAlgo import DopplerAlgo


def db2lin(x):
    """Convert dB to linear scale"""
    return 10 ** (x / 10)


def lin2db(x):
    """Convert linear scale to dB"""
    return 10 * np.log10(np.maximum(x, 1e-12))  # Protect against log(0)


class TheoreticalRadarSNR:
    """Class for calculating theoretical radar SNR using radar equation for BGT60TR13C"""
    def __init__(self):
        # Radar parameters for BGT60TR13C
        self.P_t = 10  # Transmit power in dBm (10 dBm = 10 mW)
        self.P_t_watts = db2lin(self.P_t - 30)  # Convert dBm to Watts
        
        # Antenna gains for BGT60TR13C
        self.G_tx_db = 5.0  # TX antenna gain in dBi (single antenna)
        self.G_rx_db = 5.0  # RX antenna gain in dBi (single antenna)
        self.G_tx = db2lin(self.G_tx_db)  # Convert to linear scale
        self.G_rx = db2lin(self.G_rx_db)  # Convert to linear scale
        
        # Total antenna gain (TX * RX)
        self.G_total = self.G_tx * self.G_rx
        
        # Frequency parameters
        self.f = 60.75e9  # Center frequency in Hz
        self.wavelength = constants.c / self.f  # Wavelength in meters
        
        # System parameters
        self.k = constants.k  # Boltzmann constant
        self.T = 290  # System temperature in Kelvin (room temperature)
        self.bw = 1.5e9  # Bandwidth in Hz (1.5 GHz)
        self.L_db = 2  # System losses in dB (including cable losses, etc.)
        self.L = db2lin(self.L_db)  # Convert to linear scale
        self.rcs = 1  # Radar cross section in m^2 (1 m² is a typical value for human targets)
        self.N_pulses = 128  # Number of pulses for integration

    def calculate_snr(self, range_m):
        """Calculate theoretical SNR for a given range using radar equation"""
        # Calculate signal power using radar equation with total antenna gain
        signal_power = (self.P_t_watts * self.G_total * self.wavelength**2 * self.rcs / 
                       ((4 * np.pi) ** 3 * range_m**4))
        
        # Calculate noise power
        noise_power = self.k * self.T * self.bw * self.L
        
        # Calculate SNR
        snr = signal_power / noise_power
        snr_db = lin2db(snr)
        
        # Calculate integrated SNR
        snr_integrated_db = snr_db + 10 * np.log10(self.N_pulses)
        
        return snr_db, snr_integrated_db


class MeasuredRadarSNR:
    """Class for calculating SNR from actual sensor data"""
    def __init__(self, config):
        # Radar parameters
        self.N_pulses = config.num_chirps_per_frame  # Number of pulses for integration
        self.num_samples = config.num_samples_per_chirp
        self.num_doppler_bins = 2 * config.num_chirps_per_frame
        
        # Parameters for noise floor calculation
        self.noise_floor_window = 5  # Window size for noise floor calculation
        self.noise_floor_percentile = 10  # Use 10th percentile as noise floor
        self.noise_floor_history = []  # Store noise floor history for smoothing
        self.max_history_length = 10  # Number of frames to average noise floor over
        
        # Initialize noise floor
        self.current_noise_floor_db = None

        # Add new parameters for noise spectrum
        self.noise_spectrum_history = []
        self.max_spectrum_history = 5  # Number of frames to average noise spectrum over

    def calculate_noise_floor(self, rd_spectrum):
        """Calculate real noise floor from range-doppler spectrum data using power"""
        # Convert to power (magnitude squared)
        power_spectrum = np.abs(rd_spectrum)**2
        
        # Reshape to (num_samples, num_doppler_bins, num_antennas)
        power_spectrum = power_spectrum.reshape(self.num_samples, self.num_doppler_bins, -1)
        
        # Calculate noise floor for each range bin
        noise_floor = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            # Get power values for this range bin across all doppler bins and antennas
            range_powers = power_spectrum[i, :, :].flatten()
            
            # Sort power values
            sorted_powers = np.sort(range_powers)
            
            # Use the lower percentile as noise floor
            percentile_idx = int(len(sorted_powers) * self.noise_floor_percentile / 100)
            noise_floor[i] = sorted_powers[percentile_idx]
        
        # Apply moving average to smooth noise floor
        kernel = np.ones(self.noise_floor_window) / self.noise_floor_window
        smoothed_noise_floor = np.convolve(noise_floor, kernel, mode='same')
        
        # Convert to dB
        noise_floor_db = lin2db(smoothed_noise_floor)
        
        # Update noise floor history
        self.noise_floor_history.append(noise_floor_db)
        if len(self.noise_floor_history) > self.max_history_length:
            self.noise_floor_history.pop(0)
        
        # Calculate average noise floor over history
        self.current_noise_floor_db = np.mean(self.noise_floor_history, axis=0)
        
        return self.current_noise_floor_db

    def calculate_snr_from_data(self, rd_spectrum):
        """Calculate SNR from range-doppler spectrum data using power"""
        # Convert to power (magnitude squared)
        power_spectrum = np.abs(rd_spectrum)**2
        
        # Calculate signal power from the range-doppler spectrum
        # Take the maximum value in the doppler dimension for each range bin
        signal_power = np.max(power_spectrum, axis=1)  # Shape: (num_samples, num_antennas)
        
        # Calculate average signal power across antennas
        signal_power = np.mean(signal_power, axis=1)  # Shape: (num_samples,)
        
        # Convert to dB
        signal_power_db = lin2db(signal_power)
        
        # Get real noise floor
        noise_floor_db = self.calculate_noise_floor(rd_spectrum)
        
        # Calculate SNR
        snr_db = signal_power_db - noise_floor_db
        
        # Calculate integrated SNR (coherent integration)
        snr_integrated_db = snr_db + 10 * np.log10(self.N_pulses)
        
        return snr_db, snr_integrated_db, noise_floor_db

    def calculate_noise_spectrum(self, rd_spectrum):
        """Calculate noise spectrum across doppler bins"""
        # Convert to power (magnitude squared)
        power_spectrum = np.abs(rd_spectrum)**2
        
        # Reshape to (num_samples, num_doppler_bins, num_antennas)
        power_spectrum = power_spectrum.reshape(self.num_samples, self.num_doppler_bins, -1)
        
        # Calculate noise spectrum for each doppler bin
        # We'll use the 10th percentile across range bins for each doppler bin
        noise_spectrum = np.zeros(self.num_doppler_bins)
        
        for i in range(self.num_doppler_bins):
            # Get power values for this doppler bin across all range bins and antennas
            doppler_powers = power_spectrum[:, i, :].flatten()
            
            # Sort power values
            sorted_powers = np.sort(doppler_powers)
            
            # Use the lower percentile as noise floor
            percentile_idx = int(len(sorted_powers) * self.noise_floor_percentile / 100)
            noise_spectrum[i] = sorted_powers[percentile_idx]
        
        # Convert to dB
        noise_spectrum_db = lin2db(noise_spectrum)
        
        # Update noise spectrum history
        self.noise_spectrum_history.append(noise_spectrum_db)
        if len(self.noise_spectrum_history) > self.max_spectrum_history:
            self.noise_spectrum_history.pop(0)
        
        # Calculate average noise spectrum over history
        avg_noise_spectrum_db = np.mean(self.noise_spectrum_history, axis=0)
        
        return avg_noise_spectrum_db


def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask
    return sum(1 for i in range(32) if rx_mask & (1 << i))


class LivePlot:
    def __init__(self, max_angle_degrees : float, max_range_m : float):
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        # Create figure with 6 subplots (adding input signal spectrum)
        self._fig, (self._ax_map, self._ax_theory, self._ax_meas, self._ax_noise, 
                   self._ax_spectrum, self._ax_input) = plt.subplots(nrows=6, ncols=1, figsize=(8, 36))
        self._fig.canvas.manager.set_window_title("Range-Angle Map, SNR Comparison, Noise Spectrum, and Input Signal")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True
        self.h = None
        self.h_theory = None
        self.h_meas = None
        self.h_noise = None
        self.h_spectrum = None
        self.h_input = None  # New handle for input signal spectrum

        # Add more vertical space between subplots
        self._fig.subplots_adjust(
            top=0.95,      # Top margin
            bottom=0.05,   # Bottom margin
            left=0.1,      # Left margin
            right=0.9,     # Right margin
            hspace=0.3     # Height space between subplots
        )

    def _draw_first_time(self, data : np.ndarray, theoretical_snr : np.ndarray, 
                         measured_snr : np.ndarray, noise_floor : np.ndarray, 
                         noise_spectrum : np.ndarray, input_spectrum : np.ndarray, ranges : np.ndarray):
        # Draw range-angle map
        minmin = -60
        maxmax = 0
        self.h = self._ax_map.imshow(
            data,
            vmin=minmin, vmax=maxmax,
            cmap='viridis',
            extent=(-self.max_angle_degrees, self.max_angle_degrees, 0, self.max_range_m),
            origin='lower')
        self._ax_map.set_xlabel("Angle (degrees)")
        self._ax_map.set_ylabel("Range (m)")
        self._ax_map.set_aspect("auto")
        self._ax_map.set_title("Range-Angle Map (Signal Power in dB)")

        # Draw theoretical SNR plot
        self.h_theory = self._ax_theory.plot(ranges, theoretical_snr[:, 0], 'b-', 
                                           label='Theoretical SNR (Single Chirp)')[0]
        self._ax_theory.plot(ranges, theoretical_snr[:, 1], 'r-', 
                           label='Theoretical SNR (After 128 Chirps Integration)')
        self._ax_theory.set_xlabel("Range (m)")
        self._ax_theory.set_ylabel("SNR (dB)")
        self._ax_theory.set_title("Theoretical SNR vs Range (Based on Radar Equation - Power)")
        self._ax_theory.grid(True)
        # Position legend inside plot with automatic placement
        self._ax_theory.legend(loc='best', frameon=True, framealpha=0.8)
        self._ax_theory.set_ylim([-20, 120])

        # Draw measured SNR plot
        self.h_meas = self._ax_meas.plot(ranges, measured_snr[:, 0], 'b-', 
                                       label='Measured SNR (Single Chirp)')[0]
        self._ax_meas.plot(ranges, measured_snr[:, 1], 'r-', 
                         label='Measured SNR (After 128 Chirps Integration)')
        self._ax_meas.set_xlabel("Range (m)")
        self._ax_meas.set_ylabel("SNR (dB)")
        self._ax_meas.set_title("Measured SNR vs Range (From Sensor Data - Power)")
        self._ax_meas.grid(True)
        # Position legend inside plot with automatic placement
        self._ax_meas.legend(loc='best', frameon=True, framealpha=0.8)
        self._ax_meas.set_ylim([-20, 120])

        # Draw noise floor plot with dual y-axis and auto-ranging
        ax1 = self._ax_noise
        ax2 = ax1.twinx()  # Create secondary y-axis
        
        # Plot on primary axis (dB)
        self.h_noise = ax1.plot(ranges, noise_floor, 'g-', 
                              label='System Noise Floor (10th Percentile)')[0]
        ax1.set_xlabel("Range (m)")
        ax1.set_ylabel("Power (dB)", color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True)
        
        # Auto-range the dB scale with some padding
        db_min = np.min(noise_floor) - 5  # Add 5dB padding
        db_max = np.max(noise_floor) + 5
        ax1.set_ylim([db_min, db_max])
        
        # Convert dB to µW for secondary axis
        power_uw = db2lin(noise_floor) * 1e6  # Convert to microwatts
        ax2.plot(ranges, power_uw, 'g--', alpha=0)  # Invisible line for scaling
        ax2.set_ylabel("Power (µW)", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Auto-range the µW scale
        min_uw = db2lin(db_min) * 1e6
        max_uw = db2lin(db_max) * 1e6
        ax2.set_ylim([min_uw, max_uw])
        
        # Format secondary y-axis to show reasonable numbers
        ax2.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, 
                  ['Noise Floor (dB)', f'Noise Floor ({min_uw:.1e} to {max_uw:.1e} µW)'],
                  loc='best', frameon=True, framealpha=0.8)
        
        ax1.set_title("Noise Floor vs Range (Averaged Over 10 Frames - Power)")

        # Add colorbar for range-angle map
        cbar = self._fig.colorbar(self.h, ax=self._ax_map)
        cbar.set_label('Signal Power (dB)')

        # Draw noise spectrum plot with auto-ranging
        doppler_bins = np.arange(len(noise_spectrum))
        self.h_spectrum = self._ax_spectrum.plot(doppler_bins, noise_spectrum, 'b-', 
                                               label='Noise Spectrum')[0]
        self._ax_spectrum.set_xlabel("Doppler Bin")
        self._ax_spectrum.set_ylabel("Power (dB)")
        self._ax_spectrum.set_title("Noise Spectrum Across Doppler Bins (Averaged Over 5 Frames)")
        self._ax_spectrum.grid(True)
        self._ax_spectrum.legend(loc='best', frameon=True, framealpha=0.8)
        
        # Auto-range the noise spectrum with some padding
        spectrum_min = np.min(noise_spectrum) - 5  # Add 5dB padding
        spectrum_max = np.max(noise_spectrum) + 5
        self._ax_spectrum.set_ylim([spectrum_min, spectrum_max])

        # Draw input signal spectrum plot with auto-ranging
        freq_bins = np.arange(len(input_spectrum))
        self.h_input = self._ax_input.plot(freq_bins, input_spectrum, 'b-', 
                                         label='Input Signal Spectrum')[0]
        self._ax_input.set_xlabel("Frequency Bin")
        self._ax_input.set_ylabel("Magnitude (dB)")
        self._ax_input.set_title("Input Signal Spectrum (Raw ADC Data)")
        self._ax_input.grid(True)
        self._ax_input.legend(loc='best', frameon=True, framealpha=0.8)
        
        # Auto-range the input spectrum with padding
        input_min = np.min(input_spectrum) - 5  # Add 5dB padding
        input_max = np.max(input_spectrum) + 5
        self._ax_input.set_ylim([input_min, input_max])

    def _draw_next_time(self, data : np.ndarray, theoretical_snr : np.ndarray, 
                       measured_snr : np.ndarray, noise_floor : np.ndarray, 
                       noise_spectrum : np.ndarray, input_spectrum : np.ndarray, ranges : np.ndarray):
        # Update range-angle map
        self.h.set_data(data)
        
        # Update theoretical SNR plot
        self.h_theory.set_ydata(theoretical_snr[:, 0])
        self._ax_theory.relim()
        self._ax_theory.autoscale_view()
        
        # Update measured SNR plot
        self.h_meas.set_ydata(measured_snr[:, 0])
        self._ax_meas.relim()
        self._ax_meas.autoscale_view()
        
        # Update noise floor plot with auto-ranging
        self.h_noise.set_ydata(noise_floor)
        
        # Auto-range the dB scale with padding
        db_min = np.min(noise_floor) - 5
        db_max = np.max(noise_floor) + 5
        self._ax_noise.set_ylim([db_min, db_max])
        
        # Update secondary y-axis for noise floor
        ax2 = self._ax_noise.get_shared_y_axes().get_siblings(self._ax_noise)[0]
        min_uw = db2lin(db_min) * 1e6
        max_uw = db2lin(db_max) * 1e6
        ax2.set_ylim([min_uw, max_uw])
        
        # Update noise spectrum plot with auto-ranging
        self.h_spectrum.set_ydata(noise_spectrum)
        
        # Auto-range the noise spectrum with padding
        spectrum_min = np.min(noise_spectrum) - 5
        spectrum_max = np.max(noise_spectrum) + 5
        self._ax_spectrum.set_ylim([spectrum_min, spectrum_max])

        # Update input signal spectrum plot with auto-ranging
        self.h_input.set_ydata(input_spectrum)
        
        # Auto-range the input spectrum with padding
        input_min = np.min(input_spectrum) - 5
        input_max = np.max(input_spectrum) + 5
        self._ax_input.set_ylim([input_min, input_max])

    def draw(self, data : np.ndarray, theoretical_snr : np.ndarray, 
            measured_snr : np.ndarray, noise_floor : np.ndarray, 
            noise_spectrum : np.ndarray, input_spectrum : np.ndarray, ranges : np.ndarray, title : str):
        if self._is_window_open:
            if self.h is None:
                self._draw_first_time(data, theoretical_snr, measured_snr, noise_floor, 
                                    noise_spectrum, input_spectrum, ranges)
            else:
                self._draw_next_time(data, theoretical_snr, measured_snr, noise_floor, 
                                   noise_spectrum, input_spectrum, ranges)
            self._ax_map.set_title(title)
            plt.draw()
            plt.pause(1e-3)

    def close(self, event = None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    num_beams = 27         # number of beams
    max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees

    # Calculate optimal parameters for 5-meter range
    max_range_m = 5.0  # Target maximum range
    range_resolution_m = 0.05  # Desired range resolution (5 cm)
    
    # Calculate required bandwidth for desired range resolution
    # range_resolution = c/(2*bandwidth)
    # bandwidth = c/(2*range_resolution)
    required_bandwidth = constants.c / (2 * range_resolution_m)  # ~3 GHz for 5 cm resolution
    
    # Limit bandwidth to available range (60-61.5 GHz)
    available_bandwidth = 1.5e9  # 1.5 GHz bandwidth
    center_freq = 60.75e9  # Center frequency at 60.75 GHz
    
    # Calculate chirp parameters with reduced data rate
    chirp_time = 0.0002  # 200 µs chirp time (increased from 100 µs)
    num_chirps = 128  # Reduced from 256 to lower data rate
    frame_time = 0.2  # 200 ms frame time (5 Hz frame rate)
    samples_per_chirp = 64  # Reduced from 128 to lower data rate
    
    config = Avian.DeviceConfig(
        sample_rate_Hz = 1_000_000,       # Back to 1 MHz to reduce data rate
        rx_mask = 5,                      # activate RX1 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 33,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31 (maximum)
        start_frequency_Hz = center_freq - available_bandwidth/2,  # Start frequency
        end_frequency_Hz = center_freq + available_bandwidth/2,    # End frequency
        num_chirps_per_frame = num_chirps,  # Reduced number of chirps
        num_samples_per_chirp = samples_per_chirp,  # Reduced samples per chirp
        chirp_repetition_time_s = chirp_time,  # 200 µs chirp time
        frame_repetition_time_s = frame_time,  # 200 ms frame time
        mimo_mode = 'off'                 # MIMO disabled
    )

    print("\nRadar Configuration for 5-meter range (optimized for data rate):")
    print(f"Center Frequency: {center_freq/1e9:.2f} GHz")
    print(f"Bandwidth: {available_bandwidth/1e9:.2f} GHz")
    print(f"Range Resolution: {constants.c/(2*available_bandwidth):.3f} m")
    print(f"Chirp Time: {chirp_time*1e6:.1f} µs")
    print(f"Number of Chirps: {num_chirps}")
    print(f"Frame Rate: {1/frame_time:.1f} Hz")
    print(f"Sample Rate: {config.sample_rate_Hz/1e6:.1f} MHz")
    print(f"Number of Samples per Chirp: {config.num_samples_per_chirp}")
    print(f"Maximum Range: {max_range_m:.1f} m")
    print(f"Range Resolution: {range_resolution_m:.3f} m (theoretical)")
    print(f"Data Rate: {config.num_chirps_per_frame * config.num_samples_per_chirp * 2 * 4 / frame_time / 1e6:.1f} MB/s\n")

    with Avian.Device() as device:
        # set configuration
        device.set_config(config)

        # get metrics and print them
        metrics = device.metrics_from_config(config)
        print("Device Metrics:")
        pprint.pprint(metrics)

        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)

        # Create objects for Range-Doppler, DBF, plotting, and SNR calculations
        doppler = DopplerAlgo(
            num_samples=config.num_samples_per_chirp,
            num_chirps_per_frame=config.num_chirps_per_frame,
            num_ant=num_rx_antennas
        )
        dbf = DBF(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)
        plot = LivePlot(max_angle_degrees, max_range_m)  # Updated max range
        theoretical_snr = TheoreticalRadarSNR()
        measured_snr = MeasuredRadarSNR(config)

        # Initialize SNR data arrays
        ranges = np.linspace(0.1, max_range_m, config.num_samples_per_chirp)
        theoretical_snr_data = np.zeros((config.num_samples_per_chirp, 2))  # [SNR, Integrated SNR]
        measured_snr_data = np.zeros((config.num_samples_per_chirp, 2))    # [SNR, Integrated SNR]

        # Pre-calculate theoretical SNR for all ranges
        for i, r in enumerate(ranges):
            snr_db, snr_int_db = theoretical_snr.calculate_snr(r)
            theoretical_snr_data[i, 0] = snr_db
            theoretical_snr_data[i, 1] = snr_int_db

        while not plot.is_closed():
            # frame has dimension num_rx_antennas x num_samples_per_chirp x num_chirps_per_frame
            frame = device.get_next_frame()

            rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros((config.num_samples_per_chirp, num_beams))

            for i_ant in range(num_rx_antennas): # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                mat = frame[i_ant, :, :]

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:,:,i_ant] = dfft_dbfs

            # Calculate measured SNR from sensor data
            snr_db, snr_int_db, noise_floor_db = measured_snr.calculate_snr_from_data(rd_spectrum)
            measured_snr_data[:, 0] = snr_db
            measured_snr_data[:, 1] = snr_int_db

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:,:,i_beam]
                beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Rescale map to better capture the peak
            scale = 150
            beam_range_energy = scale*(beam_range_energy/max_energy - 1)

            # Find dominant angle of target
            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]

            # Calculate noise spectrum
            noise_spectrum_db = measured_snr.calculate_noise_spectrum(rd_spectrum)
            
            # Calculate input signal spectrum from raw ADC data
            # Take the first chirp from the first antenna for visualization
            raw_signal = frame[0, :, 0]  # First antenna, all samples, first chirp
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(raw_signal))
            windowed_signal = raw_signal * window
            
            # Calculate FFT and convert to dB
            input_spectrum = np.abs(np.fft.fft(windowed_signal))
            input_spectrum_db = lin2db(input_spectrum)
            
            # Normalize to 0 dB peak
            input_spectrum_db = input_spectrum_db - np.max(input_spectrum_db)

            # Update plot with input spectrum
            plot.draw(beam_range_energy, theoretical_snr_data, measured_snr_data, 
                     noise_floor_db, noise_spectrum_db, input_spectrum_db, ranges,
                     f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")

        plot.close()

    # Create instance of TheoreticalRadarSNR
    radar_snr = TheoreticalRadarSNR()

    # Calculate SNR for distances from 0.1 to 10 meters in 0.1m steps
    distances = np.arange(0.1, 10.1, 0.1)
    snr_values = []
    snr_integrated_values = []

    print("\nTheoretical Radar SNR Values (BGT60TR13C):")
    print("Distance(m) | Single Chirp SNR(dB) | Integrated SNR(dB)")
    print("-" * 55)

    for dist in distances:
        snr_db, snr_int_db = radar_snr.calculate_snr(dist)
        print(f"{dist:9.1f} | {snr_db:17.1f} | {snr_int_db:17.1f}")
        snr_values.append(snr_db)
        snr_integrated_values.append(snr_int_db)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, snr_values, 'b-', label='Single Chirp SNR')
    plt.plot(distances, snr_integrated_values, 'r-', label='128 Chirp Integration SNR')

    plt.grid(True)
    plt.xlabel('Range (m)')
    plt.ylabel('SNR (dB)')
    plt.title('Theoretical Radar SNR vs Range for BGT60TR13C')
    plt.legend()
    plt.ylim([-20, 120])  # Set reasonable y-axis limits
    plt.xlim([0, 10])    # Set x-axis from 0 to 10 meters

    # Add text box with radar parameters
    params_text = (
        'Radar Parameters:\n'
        'Tx Power: 10 dBm\n'
        'Antenna Gain: 5 dBi\n'
        'Frequency: 60.75 GHz\n'
        'Bandwidth: 1.5 GHz\n'
        'System Loss: 2 dB\n'
        'RCS: 1 m²'
    )
    plt.text(0.02, 0.98, params_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show() 