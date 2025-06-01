import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi, Boltzmann
import pandas as pd
from datetime import datetime
import os

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

class SignalPowerCalculator:
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
        
    def calculate_theoretical_power(self, R, rcs=1.0):
        """Calculate theoretical received power for a given range and RCS"""
        # Calculate signal power using radar equation
        signal_power = (self.P_t * self.G_tx * self.G_rx * 
                       self.wavelength**2 * rcs / 
                       ((4 * pi)**3 * R**4))
        
        return {
            'rx_power': signal_power,
            'signal_power': signal_power
        }
    
    def calculate_practical_power(self, signal_data, dist_points=None):
        """Calculate practical received power from sensor data"""
        if dist_points is not None:
            # Create mask for valid range
            valid_mask = (dist_points >= self.min_range) & (dist_points <= 3.0)
            # Use only data from valid range
            valid_data = signal_data[valid_mask]
        else:
            valid_data = signal_data
            
        # Calculate signal power (using peak value from valid range)
        if dist_points is not None:
            signal_power = np.max(np.abs(valid_data))**2
        else:
            signal_power = np.max(np.abs(signal_data))**2
        
        return {
            'rx_power': signal_power,
            'signal_power': signal_power
        }

class Draw:
    def __init__(self, max_range_m, num_ant, num_samples):
        # Add FFT spectrum storage
        self._fft_buffer = []
        self._max_buffer_size = 10# Store last 10 FFT readings
        self._buffer_initialized = False
        
        self._num_ant = num_ant
        self._pln = []
        self._pln_current = []  # New line for current FFT
        self._peak_texts = []
        self._peak_line = None
        self._avg_horizontal_line = None
        self._xlim = 3.0
        self._ylim = 0.02
        self._last_data = None
        self._last_peaks = None
        self._last_peak_value = None
        self._vertical_lines = []
        self._peak_vertical_lines = []
        self._min_range = 0.2

        plt.ion()
        if num_ant == 1:
            self._fig, self._axs = plt.subplots(figsize=(6, 4))
            self._axs = [self._axs]
        else:
            self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant+1)//2, 2))
            self._axs = self._axs.flatten()

        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17/3*num_ant, 4)

        self._dist_points = np.linspace(0, max_range_m, num_samples)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._is_window_open = True
        
        self._data_saver = DataSaver()

    def _process_fft_buffer(self, data_all_ant):
        """Process FFT buffer to compute averaged spectrum"""
        if not self._buffer_initialized:
            # Initialize buffer with first reading
            for ant_idx in range(self._num_ant):
                self._fft_buffer.append([np.abs(data_all_ant[ant_idx])])
            self._buffer_initialized = True
        else:
            # Add new readings to buffer
            for ant_idx in range(self._num_ant):
                self._fft_buffer[ant_idx].append(np.abs(data_all_ant[ant_idx]))
                # Keep buffer size limited
                if len(self._fft_buffer[ant_idx]) > self._max_buffer_size:
                    self._fft_buffer[ant_idx].pop(0)

        # Compute averaged spectrum for each antenna
        averaged_data = []
        for ant_idx in range(self._num_ant):
            buffer_array = np.array(self._fft_buffer[ant_idx])
            # Simple averaging instead of multiplication
            averaged = np.mean(buffer_array, axis=0)
            averaged_data.append(averaged)

        return averaged_data

    def _draw_first_time(self, data_all_ant, averaged_data, peaks, peak_value=None):
        for ant_idx in range(self._num_ant):
            data = data_all_ant[ant_idx]
            avg_data = averaged_data[ant_idx]
            
            # Create masked data where values before 0.2m are set to 0
            masked_data = np.copy(np.abs(data))
            masked_data[self._dist_points < 0.2] = 0
            
            masked_avg_data = np.copy(avg_data)
            masked_avg_data[self._dist_points < 0.2] = 0
            
            # Calculate average only for valid range
            valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
            avg_value = np.mean(np.abs(data[valid_mask]))
            
            # Plot current FFT data
            pln_current, = self._axs[ant_idx].plot(self._dist_points, masked_data, 'r-', 
                                                  linewidth=1.0, alpha=0.5, 
                                                  label='Current FFT')
            self._pln_current.append(pln_current)
            
            # Plot averaged FFT data
            pln_avg, = self._axs[ant_idx].plot(self._dist_points, masked_avg_data, 'b-', 
                                              linewidth=1.5, 
                                              label=f'Averaged FFT (n={len(self._fft_buffer[ant_idx])})')
            self._pln.append(pln_avg)
            
            # Rest of the plotting code remains the same
            vline = self._axs[ant_idx].axvline(x=3.0, color='r', linestyle='--', alpha=0.5, 
                                              label='Range Limit')
            self._vertical_lines.append(vline)
            
            if peaks[ant_idx] is not None:
                peak_dist, peak_val = peaks[ant_idx]
                peak_vline = self._axs[ant_idx].axvline(x=peak_dist, color='g', linestyle='-', 
                                                       alpha=0.5, label='Peak Position')
                self._peak_vertical_lines.append(peak_vline)
            else:
                self._peak_vertical_lines.append(None)
            
            self._avg_horizontal_line = self._axs[ant_idx].axhline(y=avg_value, color='m', 
                                                                  linestyle=':', 
                                                                  label=f'Average: {avg_value:.2e}')
            
            if peak_value is not None:
                self._peak_line = self._axs[ant_idx].axhline(y=peak_value, color='g', 
                                                            linestyle='-', 
                                                            label=f'Peak Threshold: {peak_value:.2e}')
            
            if peaks[ant_idx] is not None:
                peak_dist, peak_val = peaks[ant_idx]
                peak_text = self._axs[ant_idx].text(peak_dist + 0.05, self._ylim * 0.95, 
                    f'Peak: {peak_val:.2e}\nDist: {peak_dist:.2f}m', 
                    fontsize=10, color='green', ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                self._peak_texts.append(peak_text)
            else:
                self._peak_texts.append(None)
            
            self._axs[ant_idx].set_xlim(0, self._xlim)
            self._axs[ant_idx].set_ylim(0, self._ylim)
            self._axs[ant_idx].grid(True)
            self._axs[ant_idx].set_xlabel('Distance (m)')
            self._axs[ant_idx].set_ylabel('Magnitude')
            self._axs[ant_idx].set_title(f'Antenna {ant_idx}')
            self._axs[ant_idx].legend(loc='upper right')

    def _draw_next_time(self, data_all_ant, averaged_data, peaks, peak_value=None):
        for ant_idx in range(self._num_ant):
            data = data_all_ant[ant_idx]
            avg_data = averaged_data[ant_idx]
            
            # Create masked data where values before 0.2m are set to 0
            masked_data = np.copy(np.abs(data))
            masked_data[self._dist_points < 0.2] = 0
            
            masked_avg_data = np.copy(avg_data)
            masked_avg_data[self._dist_points < 0.2] = 0
            
            # Calculate average only for valid range
            valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
            avg_value = np.mean(np.abs(data[valid_mask]))
            
            # Update current FFT plot
            self._pln_current[ant_idx].set_ydata(masked_data)
            self._pln_current[ant_idx].set_label('Current FFT')
            
            # Update averaged FFT plot
            self._pln[ant_idx].set_ydata(masked_avg_data)
            self._pln[ant_idx].set_label(f'Averaged FFT (n={len(self._fft_buffer[ant_idx])})')
            
            # Rest of the update code remains similar
            if self._peak_vertical_lines[ant_idx] is not None:
                self._peak_vertical_lines[ant_idx].remove()
            if peaks[ant_idx] is not None:
                peak_dist, peak_val = peaks[ant_idx]
                peak_vline = self._axs[ant_idx].axvline(x=peak_dist, color='g', linestyle='-', 
                                                       alpha=0.5, label='Peak Position')
                self._peak_vertical_lines[ant_idx] = peak_vline
            
            self._avg_horizontal_line.set_ydata([avg_value, avg_value])
            self._avg_horizontal_line.set_label(f'Average: {avg_value:.2e}')
            
            if peak_value is not None:
                self._peak_line.set_ydata([peak_value, peak_value])
                self._peak_line.set_label(f'Peak Threshold: {peak_value:.2e}')
            
            if self._peak_texts[ant_idx] is not None:
                self._peak_texts[ant_idx].remove()
            if peaks[ant_idx] is not None:
                peak_dist, peak_val = peaks[ant_idx]
                peak_text = self._axs[ant_idx].text(peak_dist + 0.05, self._ylim * 0.95,
                    f'Peak: {peak_val:.2e}\nDist: {peak_dist:.2f}m',
                    fontsize=10, color='green', ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                self._peak_texts[ant_idx] = peak_text
            
            self._axs[ant_idx].legend(loc='upper right')

    def draw(self, data_all_ant, peaks, peak_value=None):
        if not self._is_window_open:
            return
            
        # Process FFT buffer to get averaged spectrum
        averaged_data = self._process_fft_buffer(data_all_ant)
            
        # Store current data for last data point saving
        self._last_data = averaged_data
        self._last_peaks = peaks
        self._last_peak_value = peak_value
        
        if not self._pln:
            self._draw_first_time(data_all_ant, averaged_data, peaks, peak_value)
        else:
            self._draw_next_time(data_all_ant, averaged_data, peaks, peak_value)

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

    def _on_key_press(self, event):
        if event.key == 'p':  # Save key is 'p'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"radar_data_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save plot
            self._fig.savefig(f"{save_dir}/radar_plot.png", dpi=300, bbox_inches='tight')
            
            # Save FFT data
            for ant_idx in range(self._num_ant):
                if self._fft_buffer and len(self._fft_buffer[ant_idx]) > 0:
                    # Save current FFT
                    current_fft = np.abs(self._last_data[ant_idx])
                    np.save(f"{save_dir}/current_fft_antenna_{ant_idx}.npy", current_fft)
                    
                    # Save averaged FFT
                    averaged_fft = np.array(self._fft_buffer[ant_idx])
                    np.save(f"{save_dir}/averaged_fft_antenna_{ant_idx}.npy", averaged_fft)
            
            # Save all collected data to CSV
            self._data_saver.save_to_csv(save_dir)
            
            # Save last data point if available
            if self._last_data is not None and self._last_peaks is not None:
                df_last = pd.DataFrame({
                    'timestamp': [timestamp],
                    'distance': [self._last_peaks[0][0] if self._last_peaks[0] is not None else None],
                    'peak_value': [self._last_peaks[0][1] if self._last_peaks[0] is not None else None],
                    'peak_threshold': [self._last_peak_value],
                    'average_value': [np.mean(np.abs(self._last_data[0][(self._dist_points >= 0.2) & (self._dist_points <= 3.0)]))]
                })
                df_last.to_csv(f"{save_dir}/last_data_point.csv", index=False)
            
            print(f"All data saved to {save_dir}/")
            print(f"- Plot: {save_dir}/radar_plot.png")
            print(f"- Current FFT data: {save_dir}/current_fft_antenna_*.npy")
            print(f"- Averaged FFT data: {save_dir}/averaged_fft_antenna_*.npy")
            print(f"- All data: {save_dir}/radar_data.csv")
            print(f"- Last data point: {save_dir}/last_data_point.csv")

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

class DataSaver:
    def __init__(self):
        self.data_buffer = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"radar_data_{self.timestamp}"  # Changed to include timestamp
        os.makedirs(self.save_dir, exist_ok=True)
        
    def add_data(self, distance, magnitude, peak_val, peak_value, avg_value):
        self.data_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'distance': distance,
            'magnitude': magnitude,
            'peak_value': peak_val,
            'peak_threshold': peak_value,
            'average_value': avg_value
        })
        
    def save_to_csv(self, dir_path):
        if not self.data_buffer:
            return False
            
        filename = f"{dir_path}/radar_data.csv"  # Simplified filename
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(filename, index=False)
        return True
        
    def save_plot(self, fig, plot_type):
        filename = f"{self.save_dir}/radar_plot.png"  # Simplified filename
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        return True

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
        'Displays distance plot with power levels from Radar Data',
        def_nframes=1,
        def_frate=5)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        num_rx_antennas = 1  # Changed to only process antenna 0

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=3.0,
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
        power_calculator = SignalPowerCalculator()

        try:
            print("Starting continuous data acquisition. Press Ctrl+C to stop.")
            while draw.is_open():
                # Get new frame
                frame = device.get_next_frame()[0]

                data_all_ant = []
                peaks = []
                distances = []
                power_values = {'practical': None, 'theoretical': None}
                dist_points = np.linspace(0, metrics.max_range_m, chirp.num_samples)
                
                # Process only antenna 0
                samples = frame[0, :, :]
                peak_dist, data, _ = algo.compute_distance(samples)
                
                # Calculate theoretical power
                theoretical_power = power_calculator.calculate_theoretical_power(peak_dist if peak_dist >= power_calculator.min_range else power_calculator.min_range)
                
                # Calculate practical power
                practical_power = power_calculator.calculate_practical_power(data, dist_points)
                
                # Store power values
                power_values['theoretical'] = theoretical_power
                power_values['practical'] = practical_power
                
                # Mask data for peak detection
                masked_data = data.copy()
                masked_data[dist_points < power_calculator.min_range] = 0
                masked_data[dist_points > 3.0] = 0  # Additional mask for 3m limit
                valid_indices = (dist_points >= power_calculator.min_range) & (dist_points <= 3.0)
                
                # Calculate average value
                valid_mask = (dist_points >= power_calculator.min_range) & (dist_points <= 3.0)
                valid_data = np.abs(data[valid_mask])
                avg_value = np.mean(valid_data)  # Calculate average value
                
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
                
                # Calculate peak value
                peak_value = np.max(valid_data)   # Max value as peak value
                
                # Print current measurements
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print(f"Distance antenna #0: {peak_dist:^05.3f} m")
                print(f"  Peak magnitude: {peak_val:^05.2e}")
                print(f"  Peak threshold: {peak_value:^05.2e}")
                print(f"  Average value: {avg_value:^05.2e}")
                print("  " + "-"*50)
                print("Press Ctrl+C to stop")

                # Update plot
                draw.draw(data_all_ant, peaks, peak_value)
                
                # Small delay to control update rate
                plt.pause(0.1)

        except KeyboardInterrupt:
            print("\nStopping data acquisition...")
        finally:
            # Clean up
            draw.close()
            print("Application closed!")