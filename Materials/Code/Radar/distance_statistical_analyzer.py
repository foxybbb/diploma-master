import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c, pi, Boltzmann
import scipy.stats
import pandas as pd
import datetime

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
        self._num_ant = 1
        self._pln = []
        self._history_length = 100
        self._time_data = []
        self._distance_history = []
        self._current_time = 0
        self._window_size = 10
        
        # Enhanced data storage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._data_file = open(f'distance_measurements_{timestamp}.csv', 'w')
        # Enhanced header with more statistical parameters
        self._data_file.write('time,distance,mean,std,median,mode,skewness,kurtosis,q25,q75,iqr,trend,moving_avg\n')

        plt.ion()
        self._fig = plt.figure(figsize=(15, 12))
        # Create three subplots for better visualization
        self._axs1 = self._fig.add_subplot(311)  # Time history plot
        self._axs2 = self._fig.add_subplot(312)  # Distribution plot
        self._axs3 = self._fig.add_subplot(313)  # Statistical metrics plot
        
        self._fig.canvas.manager.set_window_title("Enhanced Distance Analysis - Antenna 0")
        self._fig.set_size_inches(15, 12)

        self._dist_points = np.linspace(0, max_range_m, num_samples)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _calculate_metrics(self):
        if len(self._distance_history) < 2:
            return None
        
        # Calculate basic statistics
        data = np.array(self._distance_history)
        metrics = {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'trend': np.polyfit(self._time_data, data, 1)[0]
        }
        
        # Safely calculate skewness and kurtosis only if there's enough variation
        if metrics['std'] > 1e-10:  # Only if there's meaningful variation
            try:
                metrics['skewness'] = scipy.stats.skew(data)
                metrics['kurtosis'] = scipy.stats.kurtosis(data)
            except:
                metrics['skewness'] = 0
                metrics['kurtosis'] = 0
        else:
            metrics['skewness'] = 0
            metrics['kurtosis'] = 0
        
        # Calculate mode using histogram with adaptive bins
        try:
            if metrics['std'] > 1e-10:
                hist, bin_edges = np.histogram(data, bins='auto')
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                metrics['mode'] = bin_centers[np.argmax(hist)]
            else:
                metrics['mode'] = metrics['mean']
        except:
            metrics['mode'] = metrics['mean']
        
        # Calculate moving average with proper handling of window size
        if len(data) >= self._window_size:
            ma = np.convolve(data, np.ones(self._window_size)/self._window_size, mode='valid')
            # Pad the beginning with the first valid moving average value
            pad_size = len(data) - len(ma)
            metrics['moving_avg'] = np.concatenate([np.full(pad_size, ma[0]), ma])
        else:
            metrics['moving_avg'] = np.full_like(data, metrics['mean'])
        
        # Save enhanced metrics to file
        if self._distance_history:
            self._data_file.write(
                f"{self._current_time},{self._distance_history[-1]},"
                f"{metrics['mean']},{metrics['std']},{metrics['median']},"
                f"{metrics['mode']},{metrics['skewness']},{metrics['kurtosis']},"
                f"{metrics['q25']},{metrics['q75']},{metrics['iqr']},"
                f"{metrics['trend']},{metrics['moving_avg'][-1]}\n"
            )
            self._data_file.flush()
        
        return metrics

    def _draw_first_time(self, data_all_antennas, peaks, cutoff_value=None):
        # Create mask for valid range (0.2m to 3.0m)
        valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
        data = data_all_antennas[0].copy()
        data[~valid_mask] = 0  # Zero out invalid ranges

        # Get peak distance
        peak_info = peaks[0]
        if peak_info is not None:
            peak_dist, _ = peak_info
            if 0.2 <= peak_dist <= 3.0:
                self._time_data = [self._current_time]
                self._distance_history = [peak_dist]
                self._current_time += 1

        # Plot time history
        scatter = self._axs1.scatter(self._time_data, self._distance_history, 
                                   c='blue', alpha=0.8, s=50, label='Measurements')
        self._pln.append(scatter)

        # Calculate and plot metrics if we have enough data
        metrics = self._calculate_metrics()
        if metrics is not None:
            # Plot moving average if available
            if metrics['moving_avg'] is not None:
                self._axs1.plot(self._time_data, metrics['moving_avg'], 
                              'r-', label='Moving Average', linewidth=2)
            
            # Plot trend line
            trend_line = np.poly1d(np.polyfit(self._time_data, self._distance_history, 1))
            self._axs1.plot(self._time_data, trend_line(self._time_data), 
                          'g--', label='Trend', linewidth=2)

        # Set up time history plot
        self._axs1.set_xlabel("Time (frames)")
        self._axs1.set_ylabel("Distance (m)")
        self._axs1.set_title("Distance-Time History")
        self._axs1.set_ylim(0, 3.0)
        self._axs1.set_xlim(0, self._history_length)
        self._axs1.grid(True, alpha=0.3)
        self._axs1.legend()

        # Plot distribution
        if len(self._distance_history) > 1:
            self._axs2.clear()
            hist, bins, _ = self._axs2.hist(self._distance_history, bins=20, density=True, 
                                          alpha=0.7, color='blue', label='Distribution')
            
            # Plot normal distribution fit
            x = np.linspace(0, 3.0, 100)
            if metrics is not None:
                normal_dist = np.exp(-(x - metrics['mean'])**2 / (2 * metrics['std']**2)) / (metrics['std'] * np.sqrt(2 * np.pi))
                self._axs2.plot(x, normal_dist, 'r-', label='Normal Fit', linewidth=2)
                
                # Add metrics text
                metrics_text = (f"Mean: {metrics['mean']:.3f} m\n"
                              f"Std Dev: {metrics['std']:.3f} m\n"
                              f"Median: {metrics['median']:.3f} m\n"
                              f"Mode: {metrics['mode']:.3f} m\n"
                              f"Skewness: {metrics['skewness']:.3f}\n"
                              f"Kurtosis: {metrics['kurtosis']:.3f}")
                self._axs2.text(0.02, 0.95, metrics_text, transform=self._axs2.transAxes,
                              verticalalignment='top', bbox=dict(boxstyle='round', 
                              facecolor='white', alpha=0.8))

        self._axs2.set_xlabel("Distance (m)")
        self._axs2.set_ylabel("Density")
        self._axs2.set_title("Distance Distribution")
        self._axs2.set_xlim(0, 3.0)
        self._axs2.grid(True, alpha=0.3)
        self._axs2.legend()

        # New plot for statistical metrics over time
        self._axs3.clear()
        if len(self._time_data) > 1:
            metrics_data = pd.read_csv(self._data_file.name)
            
            # Plot moving average and trend
            self._axs3.plot(metrics_data['time'], metrics_data['moving_avg'], 
                           'r-', label='Moving Average', linewidth=2)
            
            # Plot confidence intervals
            mean = metrics_data['mean']
            std = metrics_data['std']
            self._axs3.fill_between(metrics_data['time'], 
                                  mean - 2*std, mean + 2*std,
                                  color='gray', alpha=0.2, 
                                  label='95% Confidence')
            
            self._axs3.set_xlabel("Time (frames)")
            self._axs3.set_ylabel("Distance (m)")
            self._axs3.set_title("Statistical Trends")
            self._axs3.grid(True, alpha=0.3)
            self._axs3.legend()

        self._fig.tight_layout()

    def _draw_next_time(self, data_all_antennas, peaks, cutoff_value=None):
        # Create mask for valid range (0.2m to 3.0m)
        valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
        data = data_all_antennas[0].copy()
        data[~valid_mask] = 0  # Zero out invalid ranges

        # Get peak distance
        peak_info = peaks[0]
        if peak_info is not None:
            peak_dist, _ = peak_info
            if 0.2 <= peak_dist <= 3.0:
                # Update data history
                self._time_data.append(self._current_time)
                self._distance_history.append(peak_dist)
                self._current_time += 1

                # Keep only last history_length frames
                if len(self._time_data) > self._history_length:
                    self._time_data = self._time_data[-self._history_length:]
                    self._distance_history = self._distance_history[-self._history_length:]

        # Clear previous plots
        self._axs1.clear()
        self._axs2.clear()
        self._axs3.clear()

        # Plot time history
        scatter = self._axs1.scatter(self._time_data, self._distance_history,
                                   c='blue', alpha=0.8, s=50, label='Measurements')

        # Calculate and plot metrics if we have enough data
        metrics = self._calculate_metrics()
        if metrics is not None:
            # Plot moving average with proper length matching
            if len(self._distance_history) >= self._window_size:
                self._axs1.plot(self._time_data, metrics['moving_avg'], 
                              'r-', label='Moving Average', linewidth=2)
            
            # Plot trend line only if there's enough variation
            if metrics['std'] > 1e-10:
                trend_line = np.poly1d(np.polyfit(self._time_data, self._distance_history, 1))
                self._axs1.plot(self._time_data, trend_line(self._time_data), 
                              'g--', label='Trend', linewidth=2)

        # Set up time history plot
        self._axs1.set_xlabel("Time (frames)")
        self._axs1.set_ylabel("Distance (m)")
        self._axs1.set_title("Distance-Time History")
        self._axs1.set_ylim(0, 3.0)
        self._axs1.set_xlim(max(0, self._current_time - self._history_length), self._current_time)
        self._axs1.grid(True, alpha=0.3)
        self._axs1.legend()

        # Plot distribution with adaptive approach
        if len(self._distance_history) > 1:
            data = np.array(self._distance_history)
            std_dev = np.std(data)
            
            # Determine number of bins based on data variation
            if std_dev > 1e-10:
                n_bins = min(30, max(10, int(np.sqrt(len(data)))))
            else:
                n_bins = 10
            
            # Histogram
            self._axs2.clear()
            hist, bins, _ = self._axs2.hist(data, bins=n_bins, density=True, 
                                          alpha=0.6, color='blue', label='Distribution')
            
            # Only attempt KDE and normal fit if there's enough variation
            if std_dev > 1e-10:
                try:
                    # Try KDE with adjusted bandwidth
                    kde = scipy.stats.gaussian_kde(data, bw_method='silverman')
                    x_range = np.linspace(0, 3.0, 200)  # Changed to fixed range
                    self._axs2.plot(x_range, kde(x_range), 'r-', 
                                   label='KDE Fit', linewidth=2)
                except:
                    # If KDE fails, use smoothed histogram
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    smoothed_hist = np.convolve(hist, np.ones(3)/3, mode='same')
                    self._axs2.plot(bin_centers, smoothed_hist, 'r-',
                                   label='Smoothed Histogram', linewidth=2)
                
                # Try normal fit
                try:
                    mu, std = scipy.stats.norm.fit(data)
                    x_range = np.linspace(0, 3.0, 200)  # Changed to fixed range
                    normal_dist = scipy.stats.norm.pdf(x_range, mu, std)
                    self._axs2.plot(x_range, normal_dist, 'g--', 
                                   label='Normal Fit', linewidth=2)
                except:
                    pass
            
            # Add statistical information
            metrics = self._calculate_metrics()
            if metrics:
                stats_text = (
                    f"Mean: {metrics['mean']:.3f} m\n"
                    f"Median: {metrics['median']:.3f} m\n"
                    f"Mode: {metrics['mode']:.3f} m\n"
                    f"Std Dev: {metrics['std']:.3f} m\n"
                    f"IQR: {metrics['iqr']:.3f} m"
                )
                if metrics['std'] > 0:  # Only show these if there's variation
                    stats_text += f"\nSkewness: {metrics['skewness']:.3f}\n"
                    stats_text += f"Kurtosis: {metrics['kurtosis']:.3f}"
                
                self._axs2.text(0.02, 0.95, stats_text, 
                               transform=self._axs2.transAxes,
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', 
                                       facecolor='white', 
                                       alpha=0.8))

            # Set consistent x-axis limits and labels for distribution plot
            self._axs2.set_xlim(0, 3.0)
            self._axs2.set_xlabel("Distance (m)")
            self._axs2.set_ylabel("Density")
            self._axs2.set_title("Distance Distribution")
            self._axs2.grid(True, alpha=0.3)
            self._axs2.legend(loc='upper right')  # Added explicit legend position

        # New plot for statistical metrics over time
        self._axs3.clear()
        if len(self._time_data) > 1:
            metrics_data = pd.read_csv(self._data_file.name)
            
            # Plot moving average and trend
            self._axs3.plot(metrics_data['time'], metrics_data['moving_avg'], 
                           'r-', label='Moving Average', linewidth=2)
            
            # Plot confidence intervals
            mean = metrics_data['mean']
            std = metrics_data['std']
            self._axs3.fill_between(metrics_data['time'], 
                                  mean - 2*std, mean + 2*std,
                                  color='gray', alpha=0.2, 
                                  label='95% Confidence')
            
            self._axs3.set_xlabel("Time (frames)")
            self._axs3.set_ylabel("Distance (m)")
            self._axs3.set_title("Statistical Trends")
            self._axs3.grid(True, alpha=0.3)
            self._axs3.legend()

        self._fig.tight_layout()

    def draw(self, data_all_ant, peaks, cutoff_value=None):
        if not self._is_window_open:
            return
        if not self._pln:
            self._draw_first_time(data_all_ant, peaks, cutoff_value)
        else:
            self._draw_next_time(data_all_ant, peaks, cutoff_value)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self, event=None):
        if self._is_window_open:
            self._is_window_open = False
            if hasattr(self, '_data_file'):
                self._data_file.close()  # Close the file properly
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')
            print(f'Data saved to distance_measurements_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

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
    parser.add_argument('-r', '--repetition', type=int, default=128,
                        help="number of chirp repetitions, default 128")
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

        # Modify the chirp loop repetitions
        chirp_loop.loop.num_repetitions = args.repetition  # Use the command line argument
        print(f"Chirp repetitions set to: {args.repetition}")

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
                
                # Print only distance
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print(f"Distance antenna #0: {peak_dist:^05.3f} m")
                print("  " + "-"*50)
                print("Press Ctrl+C to stop")

                # Update plot
                draw.draw(data_all_ant, peaks)
                
                # Small delay to control update rate
                plt.pause(0.1)

        except KeyboardInterrupt:
            print("\nStopping data acquisition...")
        finally:
            # Clean up
            draw.close()
            print("Application closed!")