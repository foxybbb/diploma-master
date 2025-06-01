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
        self.Ns = 256      # Number of samples per chirp - increased for better range resolution
        self.fs = 2e6      # Sample rate (Hz) - increased for better range coverage
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

class ThicknessCalculator:
    def __init__(self, refractive_index=1.5):  # typical refractive index for cardboard
        self.refractive_index = refractive_index
        self.measurements = []
        self.c = 3e8  # speed of light in m/s
        
    def find_multiple_peaks(self, signal, dist_points, min_range=0.2, max_range=3.0, 
                           min_peak_distance=0.2, num_peaks=2):
        """Find multiple peaks in the signal"""
        # Create mask for valid range
        valid_mask = (dist_points >= min_range) & (dist_points <= max_range)
        valid_signal = np.abs(signal[valid_mask])
        valid_distances = dist_points[valid_mask]
        
        # Calculate signal threshold with lower sensitivity
        signal_mean = np.mean(valid_signal)
        signal_std = np.std(valid_signal)
        threshold = signal_mean + 1.0 * signal_std  # Reduced from 2.0 to 1.0 standard deviations
        
        # Find peaks above threshold
        peak_indices = []
        for i in range(1, len(valid_signal)-1):
            # Check if point is a local maximum
            is_local_max = (valid_signal[i] > valid_signal[i-1] and 
                          valid_signal[i] > valid_signal[i+1])
            
            # Check if point is significantly above local background
            local_background = np.mean(valid_signal[max(0, i-5):min(len(valid_signal), i+6)])
            is_significant = valid_signal[i] > local_background * 1.2  # 20% above local background
            
            if is_local_max and is_significant and valid_signal[i] > threshold:
                peak_indices.append(i)
        
        # Sort peaks by amplitude
        peak_indices.sort(key=lambda i: valid_signal[i], reverse=True)
        
        # Take the two strongest peaks
        peaks = []
        used_regions = set()
        
        for idx in peak_indices:
            peak_dist = valid_distances[idx]
            peak_val = valid_signal[idx]
            
            # Check if this peak is in a used region
            is_valid = True
            for used_dist in used_regions:
                if abs(peak_dist - used_dist) < min_peak_distance:
                    is_valid = False
                    break
            
            if is_valid:
                peaks.append((peak_dist, peak_val))
                used_regions.add(peak_dist)
                
                if len(peaks) >= num_peaks:
                    break
        
        return sorted(peaks, key=lambda x: x[0])  # Sort by distance
        
    def calculate_distances(self, peaks):
        """Calculate distances between all detected peaks"""
        if len(peaks) < 2:
            return None
            
        distances = []
        for i in range(len(peaks)-1):
            for j in range(i+1, len(peaks)):
                r1 = peaks[i][0]
                r2 = peaks[j][0]
                delta_r = abs(r2 - r1)
                
                # Calculate actual distance considering refractive index
                actual_distance = (self.c * delta_r) / (2 * self.refractive_index)
                
                distances.append({
                    'peak1_dist': r1,
                    'peak2_dist': r2,
                    'peak1_val': peaks[i][1],
                    'peak2_val': peaks[j][1],
                    'delta_r': delta_r,
                    'actual_distance': actual_distance
                })
        
        return distances
        
    def add_measurement(self, distance_data):
        """Add a measurement to the statistics"""
        if distance_data is not None:
            for dist in distance_data:
                self.measurements.append(dist['actual_distance'])
            
    def get_statistics(self):
        """Calculate statistics from measurements"""
        if not self.measurements:
            return None
            
        measurements = np.array(self.measurements)
        return {
            'mean': np.mean(measurements),
            'std': np.std(measurements),
            'num_samples': len(measurements)
        }

class Draw:
    def __init__(self, max_range_m, num_ant, num_samples):
        self._num_ant = num_ant
        self._pln = []
        self._peak_texts = []
        self._peak_lines = []  # Changed to list of lines
        self._avg_horizontal_line = None
        self._xlim = 3.0  # x-axis limit to 10 meters
        self._ylim = 0.03  # Static y-axis limit
        self._last_data = None
        self._last_peaks = None
        self._last_peak_value = None
        self._thickness_texts = []
        self._min_range = 0.2  # Minimum range to display

        plt.ion()
        if num_ant == 1:
            self._fig, self._axs = plt.subplots(figsize=(6, 4))
            self._axs = [self._axs]
        else:
            self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant+1)//2, 2))
            self._axs = self._axs.flatten()

        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17/3*num_ant, 4)

        # Keep full distance points
        self._dist_points = np.linspace(0, max_range_m, num_samples)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._is_window_open = True
        
        self._data_saver = DataSaver()

    def _on_key_press(self, event):
        if event.key == 'p':  # Changed save key to 'p'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"radar_data_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save plot
            self._fig.savefig(f"{save_dir}/radar_plot.png", dpi=300, bbox_inches='tight')
            
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
            print(f"- All data: {save_dir}/radar_data.csv")
            print(f"- Last data point: {save_dir}/last_data_point.csv")

    def _draw_first_time(self, data_all_ant, peaks, peak_value=None):
        for ant_idx in range(self._num_ant):
            data = data_all_ant[ant_idx]
            # Create masked data where values before 0.2m are set to 0
            masked_data = np.copy(np.abs(data))
            masked_data[self._dist_points < 0.2] = 0
            
            # Calculate average only for valid range
            valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
            avg_value = np.mean(np.abs(data[valid_mask]))
            
            # Plot masked signal
            pln_line, = self._axs[ant_idx].plot(self._dist_points, masked_data, 'b-', linewidth=1.5, label='Signal')
            self._pln.append(pln_line)
            
            # Add vertical line at 3.0m
            self._axs[ant_idx].axvline(x=3.0, color='r', linestyle='--', alpha=0.5, label='Range Limit')
            
            # Plot average value
            self._avg_horizontal_line = self._axs[ant_idx].axhline(y=avg_value, color='m', linestyle=':', label=f'Average: {avg_value:.2e}')
            
            # Plot peak threshold if available
            if peak_value is not None:
                self._peak_lines.append(self._axs[ant_idx].axhline(y=peak_value, color='g', linestyle='-', label=f'Peak Threshold: {peak_value:.2e}'))
            
            # Add markers for all peaks
            if peaks[ant_idx]:
                for i, (peak_dist, peak_val) in enumerate(peaks[ant_idx]):
                    color = 'green' if i == 0 else 'blue'  # Different colors for front/back wall
                    label = 'Front Wall' if i == 0 else 'Back Wall'
                    self._axs[ant_idx].plot(peak_dist, peak_val, 'o', color=color, 
                                          markersize=8, label=label)
                
                # Add thickness text if we have at least 2 peaks
                if len(peaks[ant_idx]) >= 2:
                    thickness_text = self._axs[ant_idx].text(
                        0.02, 0.98, '',  # Will be updated in update_thickness_text
                        transform=self._axs[ant_idx].transAxes,
                        fontsize=10, color='red',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                        verticalalignment='top')
                    self._thickness_texts.append(thickness_text)
                else:
                    self._thickness_texts.append(None)
            else:
                self._thickness_texts.append(None)
            
            # Set axis limits and labels
            self._axs[ant_idx].set_xlim(0, self._xlim)
            self._axs[ant_idx].set_ylim(0, self._ylim)
            self._axs[ant_idx].grid(True)
            self._axs[ant_idx].set_xlabel('Distance (m)')
            self._axs[ant_idx].set_ylabel('Magnitude')
            self._axs[ant_idx].set_title(f'Antenna {ant_idx}')
            self._axs[ant_idx].legend(loc='upper right')

    def _draw_next_time(self, data_all_ant, peaks, peak_value=None):
        for ant_idx in range(self._num_ant):
            # Clear the axis
            self._axs[ant_idx].clear()
            
            data = data_all_ant[ant_idx]
            # Create masked data where values before 0.2m are set to 0
            masked_data = np.copy(np.abs(data))
            masked_data[self._dist_points < 0.2] = 0
            
            # Plot main signal
            self._pln[ant_idx], = self._axs[ant_idx].plot(self._dist_points, masked_data, 'b-', 
                                                         linewidth=1.5, label='Signal')
            
            # Add markers for all peaks and distance annotations
            if peaks[ant_idx]:
                # Plot all peaks
                for i, (peak_dist, peak_val) in enumerate(peaks[ant_idx]):
                    self._axs[ant_idx].plot(peak_dist, peak_val, 'ro', 
                                          markersize=8, label=f'Peak {i+1}')
                    
                    # Add peak annotation
                    self._axs[ant_idx].annotate(f'Peak {i+1}\n{peak_dist:.3f}m',
                                              (peak_dist, peak_val),
                                              xytext=(10, 10), textcoords='offset points',
                                              bbox=dict(facecolor='white', alpha=0.7),
                                              ha='left', va='bottom')
                
                # Calculate and show distances between peaks
                if len(peaks[ant_idx]) >= 2:
                    for i in range(len(peaks[ant_idx])-1):
                        for j in range(i+1, len(peaks[ant_idx])):
                            p1 = peaks[ant_idx][i]
                            p2 = peaks[ant_idx][j]
                            mid_x = (p1[0] + p2[0]) / 2
                            mid_y = (p1[1] + p2[1]) / 2
                            delta_r = abs(p2[0] - p1[0])
                            
                            # Add distance annotation
                            self._axs[ant_idx].annotate(
                                f'Δr = {delta_r*100:.1f}cm',
                                (mid_x, mid_y),
                                xytext=(0, 20), textcoords='offset points',
                                bbox=dict(facecolor='yellow', alpha=0.5),
                                ha='center', va='bottom')
            
            # Set axis limits and labels
            self._axs[ant_idx].set_xlim(0, self._xlim)
            self._axs[ant_idx].set_ylim(0, self._ylim)
            self._axs[ant_idx].grid(True)
            self._axs[ant_idx].set_xlabel('Distance (m)')
            self._axs[ant_idx].set_ylabel('Magnitude')
            self._axs[ant_idx].set_title(f'Antenna {ant_idx}')
            
            # Update legend
            self._axs[ant_idx].legend(loc='upper right')

    def draw(self, data_all_ant, peaks, peak_value=None):
        if not self._is_window_open:
            return
            
        # Store current data for last data point saving
        self._last_data = data_all_ant
        self._last_peaks = peaks
        self._last_peak_value = peak_value
        
        if not self._pln:
            self._draw_first_time(data_all_ant, peaks, peak_value)
        else:
            self._draw_next_time(data_all_ant, peaks, peak_value)

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

    def update_thickness_text(self, ant_idx, thickness_data, statistics=None):
        """Update the thickness measurement text"""
        if self._thickness_texts[ant_idx] and thickness_data:
            text = f"Thickness: {thickness_data['thickness']*100:.1f} cm\n"
            text += f"ΔR: {thickness_data['delta_r']*100:.1f} cm"
            
            if statistics:
                text += f"\nMean: {statistics['mean']*100:.1f} cm"
                text += f"\nStd: {statistics['std']*100:.1f} cm"
                text += f"\nN: {statistics['num_samples']}"
                
            self._thickness_texts[ant_idx].set_text(text)

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
            max_range_m=3.0,  # Increased to 10 meters
            max_speed_m_s=3,
            speed_resolution_m_s=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / args.frate
        chirp_loop = sequence.loop.sub_sequence.contents
        device.sequence_from_metrics(metrics, chirp_loop)

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 2_000_000  # Increased sample rate for better range coverage
        chirp.rx_mask = (1 << num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 1000000  # Increased cutoff frequency for higher sample rate
        chirp.hp_cutoff_Hz = 80000

        device.set_acquisition_sequence(sequence)

        algo = DistanceAlgo(chirp, chirp_loop.loop.num_repetitions)
        draw = Draw(metrics.max_range_m, num_rx_antennas, chirp.num_samples)
        power_calculator = SignalPowerCalculator()
        thickness_calculator = ThicknessCalculator()

        try:
            print("Starting continuous data acquisition. Press Ctrl+C to stop.")
            while draw.is_open():
                # Get new frame
                frame = device.get_next_frame()[0]

                data_all_ant = []
                peaks = []
                distances = []
                dist_points = np.linspace(0, metrics.max_range_m, chirp.num_samples)
                
                # Process only antenna 0
                samples = frame[0, :, :]
                peak_dist, data, _ = algo.compute_distance(samples)
                
                # Find multiple peaks
                detected_peaks = thickness_calculator.find_multiple_peaks(
                    data, dist_points, min_range=0.2, max_range=3.0)
                peaks.append(detected_peaks)
                
                # Calculate distances between peaks
                distance_data = thickness_calculator.calculate_distances(detected_peaks)
                if distance_data:
                    thickness_calculator.add_measurement(distance_data)
                statistics = thickness_calculator.get_statistics()
                
                data_all_ant.append(data)
                
                # Calculate peak value for plotting
                valid_mask = (dist_points >= 0.2) & (dist_points <= 3.0)
                valid_data = np.abs(data[valid_mask])
                peak_value = np.max(valid_data)
                
                # Print current measurements
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print("Detected peaks and distances:")
                if detected_peaks:
                    print("\nPeak positions:")
                    for i, (dist, val) in enumerate(detected_peaks):
                        print(f"Peak {i+1}: {dist:.3f}m (magnitude: {val:.2e})")
                    
                    if distance_data:
                        print("\nDistances between peaks:")
                        for dist in distance_data:
                            print(f"Between peaks at {dist['peak1_dist']:.3f}m and {dist['peak2_dist']:.3f}m:")
                            print(f"  Δr = {dist['delta_r']*100:.1f}cm")
                            print(f"  Actual distance = {dist['actual_distance']*100:.1f}cm")
                        
                        if statistics:
                            print(f"\nStatistics over {statistics['num_samples']} measurements:")
                            print(f"  Mean distance: {statistics['mean']*100:.1f}cm")
                            print(f"  Std deviation: {statistics['std']*100:.1f}cm")
                else:
                    print("No peaks detected")
                
                print("\nPress Ctrl+C to stop")

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