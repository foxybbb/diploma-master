import numpy as np
import matplotlib.pyplot as plt

def calculate_snr_theory(params):
    """
    Calculate theoretical SNR for FMCW radar
    
    Parameters:
    -----------
    params : dict
        Dictionary containing radar parameters:
        - Pt: Transmit power (W)
        - Gt: Transmit antenna gain
        - Gr: Receive antenna gain
        - lambda_: Wavelength (m)
        - R: Range (m)
        - sigma: Target RCS (m²)
        - T: Temperature (K)
        - B: Bandwidth (Hz)
        - F: Noise figure
        - L: System losses
        - N_chirps: Number of chirps for integration
    
    Returns:
    --------
    snr_single : float
        SNR for single chirp
    snr_integrated : float
        SNR after chirp integration
    """
    # Extract parameters
    Pt = params['Pt']  # Transmit power (W)
    Gt = params['Gt']  # Transmit antenna gain
    Gr = params['Gr']  # Receive antenna gain
    lambda_ = params['lambda_']  # Wavelength (m)
    R = params['R']  # Range (m)
    sigma = params['sigma']  # Target RCS (m²)
    T = params['T']  # Temperature (K)
    B = params['B']  # Bandwidth (Hz)
    F = params['F']  # Noise figure
    L = params['L']  # System losses
    N_chirps = params['N_chirps']  # Number of chirps for integration
    
    # Boltzmann constant
    k = 1.380649e-23  # J/K
    
    # Calculate SNR for single chirp using radar equation
    # SNR = (Pt * Gt * Gr * lambda_^2 * sigma) / ((4π)^3 * R^4 * k * T * B * F * L)
    snr_single = (Pt * Gt * Gr * lambda_**2 * sigma) / ((4 * np.pi)**3 * R**4 * k * T * B * F * L)
    
    # Calculate SNR after integration
    # Integration gain = N_chirps (coherent integration)
    snr_integrated = snr_single * N_chirps
    
    return snr_single, snr_integrated

def plot_snr_vs_range(params, ranges):
    """
    Plot SNR vs Range for both single chirp and integrated cases
    
    Parameters:
    -----------
    params : dict
        Radar parameters (same as in calculate_snr_theory)
    ranges : np.ndarray
        Array of ranges to evaluate SNR
    """
    snr_single = []
    snr_integrated = []
    
    for R in ranges:
        params['R'] = R
        snr_s, snr_i = calculate_snr_theory(params)
        snr_single.append(10 * np.log10(snr_s))  # Convert to dB
        snr_integrated.append(10 * np.log10(snr_i))  # Convert to dB
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranges, snr_single, 'b-', label='Single Chirp')
    plt.plot(ranges, snr_integrated, 'r-', label=f'After {params["N_chirps"]} Chirp Integration')
    plt.grid(True)
    plt.xlabel('Range (m)')
    plt.ylabel('SNR (dB)')
    plt.title('Theoretical SNR vs Range')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example parameters for BGT60TR13C radar
    params = {
        'Pt': 0.1,  # 100 mW transmit power
        'Gt': 5,    # 5 dB transmit antenna gain
        'Gr': 5,    # 5 dB receive antenna gain
        'lambda_': 4.93e-3,  # Wavelength at 60.75 GHz
        'R': 1.0,   # Initial range (will be varied in plot)
        'sigma': 0.1,  # Target RCS (m²)
        'T': 290,   # Temperature (K)
        'B': 1e6,   # Bandwidth (1 MHz)
        'F': 10,    # Noise figure (dB)
        'L': 2,     # System losses (dB)
        'N_chirps': 128  # Number of chirps for integration
    }
    
    # Generate range values for plotting
    ranges = np.linspace(0.1, 5, 100)  # Ranges from 0.1m to 5m
    
    # Calculate and plot SNR
    plot_snr_vs_range(params, ranges)
    
    # Print example values at 1m range
    params['R'] = 1.0
    snr_single, snr_integrated = calculate_snr_theory(params)
    print(f"\nSNR at 1m range:")
    print(f"Single chirp SNR: {10 * np.log10(snr_single):.2f} dB")
    print(f"After {params['N_chirps']} chirp integration: {10 * np.log10(snr_integrated):.2f} dB")
    print(f"Integration gain: {10 * np.log10(params['N_chirps']):.2f} dB") 