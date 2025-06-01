import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import *

def db2lin(x):
    """Convert decibels to linear scale"""
    return 10 ** (x / 10)

def lin2db(x):
    """Convert linear scale to decibels"""
    return 10 * np.log10(x)

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
        
    def calculate_practical_power(self, signal_data, dist_points=None):
        """Calculate practical received power from sensor data"""
        if dist_points is not None:
            valid_mask = (dist_points >= self.min_range) & (dist_points <= 3.0)
            valid_data = signal_data[valid_mask]
        else:
            valid_data = signal_data
            
        signal_power = np.max(np.abs(valid_data))**2
        return {'rx_power': signal_power, 'signal_power': signal_power}

class Draw:
    def __init__(self, max_range_m, num_ant, num_samples):
        self._fft_buffer = []
        self._max_buffer_size = 10
        self._buffer_initialized = False
        self._num_ant = num_ant
        self._pln = []
        self._pln_current = []
        self._xlim = 3.0
        self._ylim = 0.02
        self._min_range = 0.2

        plt.ion()
        self._fig, self._axs = plt.subplots(figsize=(6, 4))
        self._axs = [self._axs]
        
        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17/3*num_ant, 4)
        self._dist_points = np.linspace(0, max_range_m, num_samples)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _process_fft_buffer(self, data_all_ant):
        if not self._buffer_initialized:
            for ant_idx in range(self._num_ant):
                self._fft_buffer.append([np.abs(data_all_ant[ant_idx])])
            self._buffer_initialized = True
        else:
            for ant_idx in range(self._num_ant):
                self._fft_buffer[ant_idx].append(np.abs(data_all_ant[ant_idx]))
                if len(self._fft_buffer[ant_idx]) > self._max_buffer_size:
                    self._fft_buffer[ant_idx].pop(0)

        averaged_data = []
        for ant_idx in range(self._num_ant):
            buffer_array = np.array(self._fft_buffer[ant_idx])
            averaged = np.mean(buffer_array, axis=0)
            averaged_data.append(averaged)

        return averaged_data

    def draw(self, data_all_ant, peaks, peak_value=None):
        if not self._is_window_open:
            return
            
        averaged_data = self._process_fft_buffer(data_all_ant)
        
        for ant_idx in range(self._num_ant):
            data = data_all_ant[ant_idx]
            avg_data = averaged_data[ant_idx]
            
            masked_data = np.copy(np.abs(data))
            masked_data[self._dist_points < 0.2] = 0
            
            masked_avg_data = np.copy(avg_data)
            masked_avg_data[self._dist_points < 0.2] = 0
            
            valid_mask = (self._dist_points >= 0.2) & (self._dist_points <= 3.0)
            avg_value = np.mean(np.abs(data[valid_mask]))
            
            if not self._pln:
                self._axs[ant_idx].plot(self._dist_points, masked_data, 'r-', 
                                      linewidth=1.0, alpha=0.5, label='Current FFT')
                self._axs[ant_idx].plot(self._dist_points, masked_avg_data, 'b-', 
                                      linewidth=1.5, label=f'Averaged FFT (n={len(self._fft_buffer[ant_idx])})')
                self._axs[ant_idx].axvline(x=3.0, color='r', linestyle='--', alpha=0.5, label='Range Limit')
                self._axs[ant_idx].axhline(y=avg_value, color='m', linestyle=':', label=f'Average: {avg_value:.2e}')
                
                if peaks[ant_idx] is not None:
                    peak_dist, peak_val = peaks[ant_idx]
                    self._axs[ant_idx].axvline(x=peak_dist, color='g', linestyle='-', alpha=0.5, label='Peak Position')
                    self._axs[ant_idx].text(peak_dist + 0.05, self._ylim * 0.95,
                        f'Peak: {peak_val:.2e}\nDist: {peak_dist:.2f}m',
                        fontsize=10, color='green', ha='left', va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                self._axs[ant_idx].set_xlim(0, self._xlim)
                self._axs[ant_idx].set_ylim(0, self._ylim)
                self._axs[ant_idx].grid(True)
                self._axs[ant_idx].set_xlabel('Distance (m)')
                self._axs[ant_idx].set_ylabel('Magnitude')
                self._axs[ant_idx].set_title(f'Antenna {ant_idx}')
                self._axs[ant_idx].legend(loc='upper right')
            else:
                self._axs[ant_idx].clear()
                self._axs[ant_idx].plot(self._dist_points, masked_data, 'r-', 
                                      linewidth=1.0, alpha=0.5, label='Current FFT')
                self._axs[ant_idx].plot(self._dist_points, masked_avg_data, 'b-', 
                                      linewidth=1.5, label=f'Averaged FFT (n={len(self._fft_buffer[ant_idx])})')
                if peaks[ant_idx] is not None:
                    peak_dist, peak_val = peaks[ant_idx]
                    self._axs[ant_idx].axvline(x=peak_dist, color='g', linestyle='-', alpha=0.5, label='Peak Position')
                    self._axs[ant_idx].text(peak_dist + 0.05, self._ylim * 0.95,
                        f'Peak: {peak_val:.2e}\nDist: {peak_dist:.2f}m',
                        fontsize=10, color='green', ha='left', va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                self._axs[ant_idx].set_xlim(0, self._xlim)
                self._axs[ant_idx].set_ylim(0, self._ylim)
                self._axs[ant_idx].grid(True)
                self._axs[ant_idx].set_xlabel('Distance (m)')
                self._axs[ant_idx].set_ylabel('Magnitude')
                self._axs[ant_idx].set_title(f'Antenna {ant_idx}')
                self._axs[ant_idx].legend(loc='upper right')

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

def parse_program_arguments(description, def_nframes, def_frate):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--nframes', type=int, default=def_nframes,
                        help="number of frames, default " + str(def_nframes))
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help="frame rate in Hz, default " + str(def_frate))
    return parser.parse_args()

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
                frame = device.get_next_frame()[0]
                data_all_ant = []
                peaks = []
                dist_points = np.linspace(0, metrics.max_range_m, chirp.num_samples)
                
                samples = frame[0, :, :]
                peak_dist, data, _ = algo.compute_distance(samples)
                
                masked_data = np.copy(np.abs(data))
                masked_data[dist_points < 0.2] = 0
                valid_mask = (dist_points >= 0.2) & (dist_points <= 3.0)
                valid_data = np.abs(data[valid_mask])
                
                if np.any(masked_data[valid_mask]):
                    peak_idx = np.argmax(masked_data[valid_mask])
                    peak_val = masked_data[valid_mask][peak_idx]
                    peak_dist = dist_points[valid_mask][peak_idx]
                    peaks.append((peak_dist, peak_val))
                else:
                    peaks.append(None)
                
                data_all_ant.append(data)
                peak_value = np.max(valid_data)
                avg_value = np.mean(valid_data)
                
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print(f"Distance antenna #0: {peak_dist:^05.3f} m")
                print(f"  Peak magnitude: {peak_val:^05.2e}")
                print(f"  Peak threshold: {peak_value:^05.2e}")
                print(f"  Average value: {avg_value:^05.2e}")
                print("  " + "-"*50)
                print("Press Ctrl+C to stop")

                draw.draw(data_all_ant, peaks, peak_value)
                plt.pause(0.1)

        except KeyboardInterrupt:
            print("\nStopping data acquisition...")
        finally:
            draw.close()
            print("Application closed!")