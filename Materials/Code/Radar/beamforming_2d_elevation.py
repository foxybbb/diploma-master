# ==========================================================================
# Full 2D Range-Angle Map with DBF in Azimuth and Elevation
# ===========================================================================

import pprint
import matplotlib.pyplot as plt
import numpy as np

from ifxAvian import Avian
from internal.fft_spectrum import *
from doppler import DopplerAlgo


def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask
    return sum(1 for i in range(32) if rx_mask & (1 << i))


class DBF2D:
    def __init__(self, num_rx_ant_h, num_rx_ant_v, num_beams_az, num_beams_el, max_angle_az, max_angle_el):
        self.num_beams_az = num_beams_az
        self.num_beams_el = num_beams_el
        self.max_angle_az = max_angle_az
        self.max_angle_el = max_angle_el
        self.weights = self._create_weights(num_rx_ant_h, num_rx_ant_v)

    def _create_weights(self, num_rx_ant_h, num_rx_ant_v):
        angles_az = np.linspace(-self.max_angle_az, self.max_angle_az, self.num_beams_az)
        angles_el = np.linspace(-self.max_angle_el, self.max_angle_el, self.num_beams_el)
        weights = np.zeros((self.num_beams_az, self.num_beams_el, num_rx_ant_h, num_rx_ant_v), dtype=complex)

        for i_az, angle_az in enumerate(angles_az):
            for j_el, angle_el in enumerate(angles_el):
                phase_az = np.exp(-1j * 2 * np.pi * np.arange(num_rx_ant_h) * np.sin(np.radians(angle_az)) / 2)
                phase_el = np.exp(-1j * 2 * np.pi * np.arange(num_rx_ant_v) * np.sin(np.radians(angle_el)) / 2)
                weights[i_az, j_el, :, :] = np.outer(phase_az, phase_el)

        return weights

    def run(self, rd_spectrum):  # [range, doppler, h, v]
        R, D, H, V = rd_spectrum.shape
        out = np.zeros((R, D, self.num_beams_az, self.num_beams_el), dtype=complex)
        for i in range(self.num_beams_az):
            for j in range(self.num_beams_el):
                w = self.weights[i, j, :, :]
                out[:, :, i, j] = np.einsum("rdhv,hv->rd", rd_spectrum, w)
        return out


class LivePlot2D:
    def __init__(self, max_range_m, max_angle_az, max_angle_el):
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.max_range_m = max_range_m
        self.max_angle_az = max_angle_az
        self.max_angle_el = max_angle_el
        self.is_open = True
        self.fig.canvas.manager.set_window_title("Range-Angle Map (Azimuth x Elevation)")
        self.fig.canvas.mpl_connect('close_event', self.close)

    def draw(self, energy: np.ndarray, title: str):
        # Select range bin with maximum total energy
        max_idx = np.unravel_index(np.argmax(energy), energy.shape)
        range_slice = energy[max_idx[0], :, :]
        extent = [-self.max_angle_az, self.max_angle_az, -self.max_angle_el, self.max_angle_el]
        if self.im is None:
            self.im = self.ax.imshow(range_slice.T, origin='lower', aspect='auto', extent=extent, cmap='viridis')
            self.fig.colorbar(self.im, ax=self.ax).set_label('Magnitude (a.u.)')
            self.ax.set_xlabel('Azimuth (degrees)')
            self.ax.set_ylabel('Elevation (degrees)')
        else:
            self.im.set_data(range_slice.T)
        self.ax.set_title(title)
        plt.pause(1e-3)

    def close(self, event=None):
        self.is_open = False
        plt.close('all')

    def is_closed(self):
        return not self.is_open


if __name__ == '__main__':
    num_rx_ant_h = 2
    num_rx_ant_v = 2
    num_beams_az = 27
    num_beams_el = 15
    max_angle_az = 45
    max_angle_el = 30

    config = Avian.DeviceConfig(
        sample_rate_Hz=1_000_000,
        rx_mask=5,  # 4 RX antennas 0b1111
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

        doppler = DopplerAlgo(config, num_rx_antennas_from_config(config))
        dbf2d = DBF2D(num_rx_ant_h, num_rx_ant_v, num_beams_az, num_beams_el, max_angle_az, max_angle_el)
        plot = LivePlot2D(metrics.max_range_m, max_angle_az, max_angle_el)

        while not plot.is_closed():
            frame = device.get_next_frame()
            rd_spectrum = np.zeros((config.num_samples_per_chirp, config.num_chirps_per_frame, num_rx_ant_h, num_rx_ant_v), dtype=complex)

            for h in range(num_rx_ant_h):
                for v in range(num_rx_ant_v):
                    idx = h * num_rx_ant_v + v
                    mat = frame[idx, :, :]
                    rd_spectrum[:, :, h, v] = doppler.compute_doppler_map(mat, idx)

            rd_beam_formed = dbf2d.run(rd_spectrum)
            beam_range_energy = np.linalg.norm(rd_beam_formed, axis=1)  # collapse Doppler axis
            beam_range_energy = 150 * (beam_range_energy / np.max(beam_range_energy) - 1)

            plot.draw(beam_range_energy, "2D Range-Angle Map (Azimuth x Elevation)")

        plot.close()