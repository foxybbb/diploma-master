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

import argparse
import matplotlib.pyplot as plt
import numpy as np

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DopplerAlgo import *


def parse_program_arguments(description, def_nframes, def_frate):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--nframes', type=int, default=def_nframes,
                        help=f"number of frames, default {def_nframes}")
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help=f"frame rate in Hz, default {def_frate}")
    return parser.parse_args()


def linear_to_dB(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))  # защита от log(0)


class Draw:
    def __init__(self, max_speed_m_s, max_range_m, num_ant):
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant
        plt.ion()
        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        self._ax = [ax] if num_ant == 1 else ax
        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.set_size_inches(3 * num_ant + 1, 3 + 1 / num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])
        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            h = self._ax[i_ant].imshow(
                data, vmin=minmin, vmax=maxmax, cmap='viridis',
                extent=(-self._max_speed_m_s, self._max_speed_m_s, 0, self._max_range_m),
                origin='lower')
            self._h.append(h)
            self._ax[i_ant].set_xlabel("velocity (m/s)")
            self._ax[i_ant].set_ylabel("distance (m)")
            self._ax[i_ant].set_title("antenna #" + str(i_ant))
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])
        cbar = self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")

    def _draw_next_time(self, data_all_antennas):
        for i_ant in range(self._num_ant):
            self._h[i_ant].set_data(data_all_antennas[i_ant])

    def draw(self, data_all_antennas):
        if self._is_window_open:
            if len(self._h) == 0:
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_open(self):
        return self._is_window_open


if __name__ == '__main__':
    args = parse_program_arguments("""Displays range doppler map from Radar Data""", def_nframes=50, def_frate=5)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]

        metrics = FmcwMetrics(
            range_resolution_m=0.15,
            max_range_m=4.8,
            max_speed_m_s=2.45,
            speed_resolution_m_s=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / args.frate

        chirp_loop = sequence.loop.sub_sequence.contents
        device.sequence_from_metrics(metrics, chirp_loop)

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 1_000_000
        chirp.rx_mask = (1 << num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500000
        chirp.hp_cutoff_Hz = 80000

        device.set_acquisition_sequence(sequence)

        doppler = DopplerAlgo(chirp.num_samples, chirp_loop.loop.num_repetitions, num_rx_antennas)
        draw = Draw(metrics.max_speed_m_s, metrics.max_range_m, num_rx_antennas)

        mti_alpha = 0.4
        empty_dmap = doppler.compute_doppler_map(np.zeros((chirp_loop.loop.num_repetitions, chirp.num_samples)), 0)
        mti_history = [np.zeros_like(np.abs(empty_dmap)) for _ in range(num_rx_antennas)]

        for frame_number in range(args.nframes):
            if not draw.is_open():
                break
            frame_contents = device.get_next_frame()
            frame_data = frame_contents[0]
            data_all_antennas = []
            for i_ant in range(num_rx_antennas):
                mat = frame_data[i_ant, :, :]
                mat = mat - np.mean(mat)
                doppler_map = doppler.compute_doppler_map(mat, i_ant)
                doppler_map_amp = np.abs(doppler_map)

                # --- Apply 2D MTI on amplitude ---
                mti_avg = mti_history[i_ant]
                doppler_map_mti = doppler_map_amp - mti_alpha * mti_avg
                mti_history[i_ant] = mti_alpha * doppler_map_amp + (1 - mti_alpha) * mti_avg
                # ---------------------------------

                dfft_dbfs = linear_to_dB(doppler_map_mti)
                data_all_antennas.append(dfft_dbfs)
            # --- Threshold suppression ---
            # threshold_db = -60
            # data_all_antennas = [np.where(data > threshold_db, data, threshold_db) for data in data_all_antennas]
            draw.draw(data_all_antennas)

        draw.close()
