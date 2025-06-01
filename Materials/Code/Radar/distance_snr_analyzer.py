import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi, Boltzmann

def db2lin(x):
    """Convert decibels to linear scale"""
    return 10 ** (x / 10)

def lin2db(x):
    """Convert linear scale to decibels"""
    return 10 * np.log10(x)

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import *

class SNRCalculator:
    def __init__(self):
        # Hardware parameters
        self.fc = 60.75e9  # Center frequency (Hz)
        self.B = 1.5e9     # FMCW bandwidth (Hz)
        self.tau = 200e-6  # Chirp duration (s)
        self.Nc = 128      # Number of chirps per frame
        self.Ns = 64       # Number of samples per chirp
        self.fs = 1e6      # Sample rate (Hz)
        self.min_range = 0.2  # Minimum range to consider (m)
        
        # Antenna parameters
        self.G_tx_db = 5   # TX antenna gain (dBi)
        self.G_rx_db = 5   # RX antenna gain (dBi)
        self.G_tx = db2lin(self.G_tx_db)
        self.G_rx = db2lin(self.G_rx_db)
        
        # Other parameters
        self.P_t = 1.4e6   # Transmit power (W)
        self.wavelength = c / self.fc  # Wavelength (m)
        self.k = Boltzmann  # Boltzmann constant
        self.T = 290       # System temperature (K)
        self.L_db = 8      # System losses (dB)
        self.L = db2lin(self.L_db)
        
        # Calculate noise bandwidth
        self.bw = self.fs / 2  # Noise bandwidth (Hz)
        
        # Calculate system noise floor
        self.system_noise_floor = self.calculate_system_noise_floor()
        
    def calculate_system_noise_floor(self):
        """Calculate the system noise floor in dBm"""
        # Thermal noise power in dBm
        thermal_noise_db = lin2db(self.k * self.T * self.bw) + 30  # Convert to dBm
        # Add system losses
        system_noise_floor_db = thermal_noise_db + self.L_db
        return system_noise_floor_db
        
    def calculate_theoretical_snr(self, R, rcs=1.0):
        """Calculate theoretical SNR for a given range and RCS"""
        # Calculate signal power
        signal_power = (self.P_t * self.G_tx * self.G_rx * 
                       self.wavelength**2 * rcs / 
                       ((4 * pi)**3 * R**4))
        
        # Calculate noise power
        noise_power = self.k * self.T * self.bw * self.L
        
        # Calculate SNR
        snr = signal_power / noise_power
        snr_db = lin2db(snr)
        
        # Calculate integrated SNR for Nc chirps
        snr_integrated_db = snr_db + 10 * np.log10(self.Nc)
        
        # Calculate received power in dBm
        rx_power_db = lin2db(signal_power) + 30  # Convert to dBm
        
        return {
            'single_chirp_snr_db': snr_db,
            'integrated_snr_db': snr_integrated_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'rx_power_db': rx_power_db,
            'system_noise_floor_db': self.system_noise_floor
        }
    
    def calculate_practical_snr(self, signal_data, dist_points=None):
        """Calculate practical SNR from sensor data, ignoring data below min_range"""
        if dist_points is not None:
            # Create mask for valid range
            valid_mask = dist_points >= self.min_range
            # Use only data from valid range for noise floor calculation
            valid_data = signal_data[valid_mask]
        else:
            valid_data = signal_data
            
        # Estimate noise floor from the lower 10% of valid data
        noise_floor = np.mean(np.abs(valid_data[valid_data < np.percentile(valid_data, 10)]))
        
        # Calculate signal power (using peak value from valid range)
        if dist_points is not None:
            signal_power = np.max(np.abs(valid_data))**2
        else:
            signal_power = np.max(np.abs(signal_data))**2
        
        # Calculate noise power
        noise_power = noise_floor**2
        
        # Calculate SNR
        snr = signal_power / noise_power
        snr_db = lin2db(snr)
        
        # Calculate integrated SNR for Nc chirps
        snr_integrated_db = snr_db + 10 * np.log10(self.Nc)
        
        # Calculate received power in dBm
        rx_power_db = lin2db(signal_power) + 30  # Convert to dBm
        
        return {
            'single_chirp_snr_db': snr_db,
            'integrated_snr_db': snr_integrated_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'rx_power_db': rx_power_db,
            'measured_noise_floor_db': lin2db(noise_power) + 30  # Convert to dBm
        }

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

class NoiseAnalyzer:
    def __init__(self):
        self._fig_noise = None
        self._axs_noise = None
        self._is_window_open = True
        
    def setup_plots(self, num_ant):
        plt.ion()
        self._fig_noise, self._axs_noise = plt.subplots(nrows=2, ncols=num_ant, figsize=(17/3 * num_ant, 8))
        self._fig_noise.canvas.manager.set_window_title("Noise Analysis")
        self._fig_noise.canvas.mpl_connect('close_event', self.close)
        
        # Set titles for all subplots
        for i in range(num_ant):
            self._axs_noise[0, i].set_title(f"Noise Signal - Antenna #{i}")
            self._axs_noise[0, i].set_xlabel("Sample Index")
            self._axs_noise[0, i].set_ylabel("Amplitude")
            
            self._axs_noise[1, i].set_title(f"Noise Spectrum - Antenna #{i}")
            self._axs_noise[1, i].set_xlabel("Frequency (Hz)")
            self._axs_noise[1, i].set_ylabel("Magnitude (dB)")
        
        self._fig_noise.tight_layout()
        
    def update_plots(self, data_all_antennas, fs=1e6):
        if not self._is_window_open:
            return
            
        if self._fig_noise is None:
            self.setup_plots(len(data_all_antennas))
            
        for i, data in enumerate(data_all_antennas):
            # Clear previous plots
            self._axs_noise[0, i].clear()
            self._axs_noise[1, i].clear()
            
            # Plot time domain signal
            time = np.arange(len(data)) / fs
            self._axs_noise[0, i].plot(time, data)
            self._axs_noise[0, i].set_title(f"Noise Signal - Antenna #{i}")
            self._axs_noise[0, i].set_xlabel("Time (s)")
            self._axs_noise[0, i].set_ylabel("Amplitude")
            
            # Calculate and plot spectrum
            spectrum = np.abs(np.fft.fft(data))
            freq = np.fft.fftfreq(len(data), 1/fs)
            # Plot only positive frequencies
            pos_freq_mask = freq >= 0
            self._axs_noise[1, i].plot(freq[pos_freq_mask], lin2db(spectrum[pos_freq_mask]))
            self._axs_noise[1, i].set_title(f"Noise Spectrum - Antenna #{i}")
            self._axs_noise[1, i].set_xlabel("Frequency (Hz)")
            self._axs_noise[1, i].set_ylabel("Magnitude (dB)")
            
            # Add grid for better readability
            self._axs_noise[0, i].grid(True)
            self._axs_noise[1, i].grid(True)
            
        self._fig_noise.tight_layout()
        self._fig_noise.canvas.draw_idle()
        self._fig_noise.canvas.flush_events()
        
    def close(self, event=None):
        if self._is_window_open:
            self._is_window_open = False
            plt.close(self._fig_noise)
            print('Noise analysis window closed!')
            
    def is_open(self):
        return self._is_window_open

class SNRPlotter:
    def __init__(self):
        self._fig_snr = None
        self._axs_snr = None
        self._is_window_open = True
        
    def setup_plots(self, num_ant):
        plt.ion()
        self._fig_snr, self._axs_snr = plt.subplots(nrows=3, ncols=num_ant, figsize=(17/3 * num_ant, 12))
        self._fig_snr.canvas.manager.set_window_title("SNR and Range Analysis")
        self._fig_snr.canvas.mpl_connect('close_event', self.close)
        
        # Set titles for all subplots
        for i in range(num_ant):
            # Single chirp SNR plot
            self._axs_snr[0, i].set_title(f"Single Chirp SNR - Antenna #{i}")
            self._axs_snr[0, i].set_xlabel("Distance (m)")
            self._axs_snr[0, i].set_ylabel("SNR (dB)")
            self._axs_snr[0, i].grid(True)
            
            # Integrated SNR plot
            self._axs_snr[1, i].set_title(f"Integrated SNR (128 chirps) - Antenna #{i}")
            self._axs_snr[1, i].set_xlabel("Distance (m)")
            self._axs_snr[1, i].set_ylabel("SNR (dB)")
            self._axs_snr[1, i].grid(True)
            
            # Received Power plot
            self._axs_snr[2, i].set_title(f"Received Power - Antenna #{i}")
            self._axs_snr[2, i].set_xlabel("Distance (m)")
            self._axs_snr[2, i].set_ylabel("Power (dBm)")
            self._axs_snr[2, i].grid(True)
        
        self._fig_snr.tight_layout()
        
    def update_plots(self, distances, practical_snr_values, theoretical_snr_values, num_ant):
        if not self._is_window_open:
            return
            
        if self._fig_snr is None:
            self.setup_plots(num_ant)
        
        # Create distance points for theoretical curves, starting from min_range
        dist_points = np.linspace(snr_calculator.min_range, 5.0, 100)  # From 0.2m to 5.0m
        
        for i in range(num_ant):
            # Clear previous plots
            for row in range(3):
                self._axs_snr[row, i].clear()
                self._axs_snr[row, i].grid(True)
            
            # Calculate theoretical values for the distance range
            theoretical_snr_single = []
            theoretical_snr_integrated = []
            theoretical_rx_power = []
            for d in dist_points:
                snr = snr_calculator.calculate_theoretical_snr(d)
                theoretical_snr_single.append(snr['single_chirp_snr_db'])
                theoretical_snr_integrated.append(snr['integrated_snr_db'])
                theoretical_rx_power.append(snr['rx_power_db'])
            
            # Plot single chirp SNR
            self._axs_snr[0, i].plot(dist_points, theoretical_snr_single, 'b--', 
                                    label='Theoretical', alpha=0.7)
            if distances[i] >= snr_calculator.min_range:  # Only plot if distance is valid
                self._axs_snr[0, i].plot(distances[i], practical_snr_values[i]['single_chirp_snr_db'], 
                                        'ro', label='Practical', markersize=8)
            self._axs_snr[0, i].axhline(y=0, color='r', linestyle=':', label='Detection Threshold')
            self._axs_snr[0, i].set_title(f"Single Chirp SNR - Antenna #{i}")
            self._axs_snr[0, i].set_xlabel("Distance (m)")
            self._axs_snr[0, i].set_ylabel("SNR (dB)")
            self._axs_snr[0, i].legend()
            
            # Plot integrated SNR
            self._axs_snr[1, i].plot(dist_points, theoretical_snr_integrated, 'b--', 
                                    label='Theoretical', alpha=0.7)
            if distances[i] >= snr_calculator.min_range:  # Only plot if distance is valid
                self._axs_snr[1, i].plot(distances[i], practical_snr_values[i]['integrated_snr_db'], 
                                        'ro', label='Practical', markersize=8)
            self._axs_snr[1, i].axhline(y=0, color='r', linestyle=':', label='Detection Threshold')
            self._axs_snr[1, i].set_title(f"Integrated SNR (128 chirps) - Antenna #{i}")
            self._axs_snr[1, i].set_xlabel("Distance (m)")
            self._axs_snr[1, i].set_ylabel("SNR (dB)")
            self._axs_snr[1, i].legend()
            
            # Plot received power
            self._axs_snr[2, i].plot(dist_points, theoretical_rx_power, 'b--', 
                                    label='Theoretical Rx Power', alpha=0.7)
            if distances[i] >= snr_calculator.min_range:  # Only plot if distance is valid
                self._axs_snr[2, i].plot(distances[i], practical_snr_values[i]['rx_power_db'], 
                                        'ro', label='Measured Rx Power', markersize=8)
            # Add system noise floor line
            self._axs_snr[2, i].axhline(y=theoretical_snr_values[i]['system_noise_floor_db'], 
                                       color='g', linestyle='--', 
                                       label='System Noise Floor')
            if 'measured_noise_floor_db' in practical_snr_values[i]:
                self._axs_snr[2, i].axhline(y=practical_snr_values[i]['measured_noise_floor_db'], 
                                           color='m', linestyle='--', 
                                           label='Measured Noise Floor')
            self._axs_snr[2, i].set_title(f"Received Power - Antenna #{i}")
            self._axs_snr[2, i].set_xlabel("Distance (m)")
            self._axs_snr[2, i].set_ylabel("Power (dBm)")
            self._axs_snr[2, i].legend()
            
        self._fig_snr.tight_layout()
        self._fig_snr.canvas.draw_idle()
        self._fig_snr.canvas.flush_events()
        
    def close(self, event=None):
        if self._is_window_open:
            self._is_window_open = False
            plt.close(self._fig_snr)
            print('SNR analysis window closed!')
            
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
        def_nframes=1,
        def_frate=5)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=5.0,
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
        noise_analyzer = NoiseAnalyzer()
        snr_plotter = SNRPlotter()
        snr_calculator = SNRCalculator()

        # Get single frame
        frame = device.get_next_frame()[0]

        data_all_ant = []
        peaks = []
        distances = []
        practical_snr_values = []
        theoretical_snr_values = []
        dist_points = np.linspace(0, metrics.max_range_m, chirp.num_samples)
        
        for i in range(num_rx_antennas):
            samples = frame[i, :, :]
            peak_dist, data, snr_db = algo.compute_distance(samples)
            
            # Calculate theoretical SNR
            theoretical_snr = snr_calculator.calculate_theoretical_snr(peak_dist if peak_dist >= snr_calculator.min_range else snr_calculator.min_range)
            
            # Calculate practical SNR from the data, passing distance points for masking
            practical_snr = snr_calculator.calculate_practical_snr(data, dist_points)
            
            # Store values for plotting
            distances.append(peak_dist)
            practical_snr_values.append(practical_snr)
            theoretical_snr_values.append(theoretical_snr)
            
            # Mask data for peak detection
            masked_data = data.copy()
            masked_data[dist_points < snr_calculator.min_range] = 0
            valid_indices = dist_points >= snr_calculator.min_range
            if np.any(masked_data[valid_indices]):
                peak_idx = np.argmax(masked_data[valid_indices])
                peak_val = masked_data[valid_indices][peak_idx]
                peak_dist = dist_points[valid_indices][peak_idx]
                peaks.append((peak_dist, peak_val))
            else:
                peaks.append(None)
                peak_dist = 0
                peak_val = 0
            data_all_ant.append(data)
            
            print(f"Distance antenna #{i}: {peak_dist:^05.3f} m")
            print(f"  Peak magnitude: {peak_val:^05.3f}")
            print(f"  Practical SNR (single chirp): {practical_snr['single_chirp_snr_db']:^05.1f} dB")
            print(f"  Practical SNR (integrated): {practical_snr['integrated_snr_db']:^05.1f} dB")
            if peak_dist >= snr_calculator.min_range:
                print(f"  Theoretical SNR (single chirp): {theoretical_snr['single_chirp_snr_db']:^05.1f} dB")
                print(f"  Theoretical SNR (integrated): {theoretical_snr['integrated_snr_db']:^05.1f} dB")
            print("  " + "-"*50)

        # Draw all plots
        draw.draw(data_all_ant, peaks)
        noise_analyzer.update_plots(data_all_ant, fs=chirp.sample_rate_Hz)
        snr_plotter.update_plots(distances, practical_snr_values, theoretical_snr_values, num_rx_antennas)
        
        # Keep plots open
        plt.show(block=True)
        
        # Clean up
        draw.close()
        noise_analyzer.close()
        snr_plotter.close()