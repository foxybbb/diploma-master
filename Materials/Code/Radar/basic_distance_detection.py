import argparse
import numpy as np
import matplotlib.pyplot as plt

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import *

# -------------------------------------------------
# Presentation with peak markers
# -------------------------------------------------
class Draw:
    # Рисует графики для каждого приемника и отмечает пики
    def __init__(self, max_range_m, num_ant, num_samples):
        self._num_ant = num_ant
        self._pln = []
        self._markers = []  # Store vertical lines
        self._peak_texts = []  # Store text annotations for peak values

        plt.ion()
        self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        self._fig.canvas.manager.set_window_title("Range FFT with Peaks")
        self._fig.set_size_inches(17 / 3 * num_ant, 4)

        self._dist_points = np.linspace(0, max_range_m, num_samples)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas, peaks):
        # Первичная отрисовка
        minmin = min(np.min(data) for data in data_all_antennas)
        maxmax = max(np.max(data) for data in data_all_antennas)

        for i in range(self._num_ant):
            ax = self._axs[i] if isinstance(self._axs, np.ndarray) else self._axs
            data = data_all_antennas[i]
            # Основной график (без маскировки 0-0.2 м)
            pln_line, = ax.plot(self._dist_points, data)
            ax.set_ylim(minmin, 1.1 * maxmax)  # Margin for high values
            self._pln.append(pln_line)

            # Вертикальная линия и текст только для валидных пиков
            peak_info = peaks[i]
            if peak_info is not None:
                peak_dist, peak_val = peak_info
                marker = ax.axvline(peak_dist, color='r', linestyle='--')  # Vertical dashed line
                # Текст с точным значением пика
                peak_text = ax.text(peak_dist + 0.05, 0.95 * (1.1 * maxmax), f'{peak_val:.4f}',
                                    fontsize=9, color='red', ha='left', va='top',
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                marker = None  # No line if no valid peak
                peak_text = ax.text(0, 0, '', visible=False)  # Placeholder text
            self._markers.append(marker)
            self._peak_texts.append(peak_text)

            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("FFT magnitude")
            ax.set_title(f"Antenna #{i}")

        self._fig.tight_layout()

    def _draw_next_time(self, data_all_antennas, peaks):
        # Обновление данных и маркеров
        maxmax = max(np.max(data) for data in data_all_antennas)
        minmin = min(np.min(data) for data in data_all_antennas)
        for i in range(self._num_ant):
            ax = self._axs[i] if isinstance(self._axs, np.ndarray) else self._axs
            data = data_all_antennas[i]
            # Обновляем данные без маскировки
            self._pln[i].set_ydata(data)
            ax.set_ylim(minmin, 1.1 * maxmax)  # Update y-axis

            # Обновляем вертикальную линию и текст
            if self._markers[i] is not None:
                self._markers[i].remove()
            self._peak_texts[i].set_visible(False)
            peak_info = peaks[i]
            if peak_info is not None:
                peak_dist, peak_val = peak_info
                self._markers[i] = ax.axvline(peak_dist, color='r', linestyle='--')
                self._peak_texts[i].set_position((peak_dist + 0.05, 0.95 * (1.1 * maxmax)))
                self._peak_texts[i].set_text(f'{peak_val:.4f}')
                self._peak_texts[i].set_visible(True)
            else:
                self._markers[i] = None

    def draw(self, data_all_antennas, peaks):
        # Отрисовать кадр с пиками
        if not self._is_window_open:
            return
        if not self._pln:
            self._draw_first_time(data_all_antennas, peaks)
        else:
            self._draw_next_time(data_all_antennas, peaks)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self, event=None):
        if self._is_window_open:
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_open(self):
        return self._is_window_open

# -------------------------------------------------
# Парсер аргументов
# -------------------------------------------------
def parse_program_arguments(description, def_nframes, def_frate):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--nframes', type=int, default=def_nframes,
                        help="number of frames, default " + str(def_nframes))
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help="frame rate in Hz, default " + str(def_frate))
    return parser.parse_args()

# -------------------------------------------------
# Основная логика
# -------------------------------------------------
if __name__ == '__main__':
    args = parse_program_arguments(
        'Displays distance plot with peak markers from Radar Data',
        def_nframes=50,
        def_frate=5)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=1.6,
            max_speed_m_s=3,
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

        algo = DistanceAlgo(chirp, chirp_loop.loop.num_repetitions)
        draw = Draw(metrics.max_range_m, num_rx_antennas, chirp.num_samples)

        for _ in range(args.nframes):
            if not draw.is_open():
                break
            frame = device.get_next_frame()[0]

            data_all_ant = []
            peaks = []
            dist_points = np.linspace(0, metrics.max_range_m, chirp.num_samples)
            for i in range(num_rx_antennas):
                samples = frame[i, :, :]
                peak_dist, data, snr_db = algo.compute_distance(samples)
                # Маскируем данные только для вычисления пика
                masked_data = data.copy()
                masked_data[dist_points < 0.2] = 0
                # Находим пик только в валидной зоне (>= 0.2 м)
                valid_indices = dist_points >= 0.2
                if np.any(masked_data[valid_indices]):  # Проверяем, есть ли ненулевые значения
                    peak_idx = np.argmax(masked_data[valid_indices])
                    peak_val = masked_data[valid_indices][peak_idx]
                    peak_dist = dist_points[valid_indices][peak_idx]
                    peaks.append((peak_dist, peak_val))
                else:
                    peaks.append(None)  # Нет валидного пика
                    peak_dist = 0
                    peak_val = 0
                data_all_ant.append(data)  # Полные данные для отображения
                print(f"Distance antenna #{i}: {peak_dist:^05.3f} m, Peak magnitude: {peak_val:^05.3f}, SNR: {snr_db:^05.1f} dB")

            draw.draw(data_all_ant, peaks)

        draw.close()