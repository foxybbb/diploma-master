# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
# ===========================================================================

import pprint
import matplotlib.pyplot as plt
import numpy as np

from ifxAvian import Avian
from internal.fft_spectrum import *
from helpers.DBF import DBF
from doppler import DopplerAlgo


def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask
    return sum(1 for i in range(32) if rx_mask & (1 << i))


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float):
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m
        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)
        self._fig.canvas.manager.set_window_title("Range-Angle-Map using DBF")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data: np.ndarray):
        minmin = -60
        maxmax = 0
        self.h = self._ax.imshow(
            data, vmin=minmin, vmax=maxmax, cmap='viridis',
            extent=(-self.max_angle_degrees, self.max_angle_degrees, 0, self.max_range_m),
            origin='lower'
        )
        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])
        cbar = self._fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        self.h.set_data(data)

    def draw(self, data: np.ndarray, title: str):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)
            plt.draw()
            plt.pause(1e-3)

    def close(self, event=None):
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
    num_beams = 27
    max_angle_degrees = 45

    config = Avian.DeviceConfig(
        sample_rate_Hz=1_000_000,
        rx_mask=5,
        tx_mask=1,
        if_gain_dB=33,
        tx_power_level=31,
        start_frequency_Hz=60e9,
        end_frequency_Hz=61.5e9,
        num_chirps_per_frame=64,
        num_samples_per_chirp=64,
        chirp_repetition_time_s=0.0005,
        frame_repetition_time_s=0.15,
        mimo_mode='off'
    )

    with Avian.Device() as device:
        device.set_config(config)
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)
        max_range_m = metrics.max_range_m

        num_rx_antennas = num_rx_antennas_from_config(
            config)
        doppler = DopplerAlgo(config, num_rx_antennas)
        dbf = DBF(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)
        plot = LivePlot(max_angle_degrees, max_range_m)

        while not plot.is_closed():
            frame = device.get_next_frame()
            rd_spectrum = np.zeros((config.num_samples_per_chirp, 2 * config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            for i_ant in range(num_rx_antennas):
                mat = frame[i_ant, :, :]
                rd_spectrum[:, :, i_ant] = doppler.compute_doppler_map(mat, i_ant)

            rd_beam_formed = dbf.run(rd_spectrum)
            beam_range_energy = np.zeros((config.num_samples_per_chirp, num_beams))

            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]
                beam_range_energy[:, i_beam] = np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            max_energy = np.max(beam_range_energy)
            scale = 150
            beam_range_energy = scale * (beam_range_energy / max_energy - 1)

            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]

            plot.draw(beam_range_energy, f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")

        plot.close()
