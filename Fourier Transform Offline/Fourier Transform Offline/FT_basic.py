import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    dt = sampled_times[1] - sampled_times[0] 
    
    for i, freq in enumerate(frequencies):
        integrand_real = signal * np.cos(-2 * np.pi * freq * sampled_times)
        integrand_imag = signal * np.sin(-2 * np.pi * freq * sampled_times)
        
        ft_result_real[i] = np.trapz(integrand_real, dx=dt)
        ft_result_imag[i] = np.trapz(integrand_imag, dx=dt)

    return ft_result_real, ft_result_imag

def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    df = frequencies[1] - frequencies[0]
    
    ft_real, ft_imag = ft_signal
    
    for i, t in enumerate(sampled_times):
        integrand_real = (ft_real * np.cos(2 * np.pi * frequencies * t) - 
                         ft_imag * np.sin(2 * np.pi * frequencies * t))
        reconstructed_signal[i] = np.trapz(integrand_real, dx=df)
        
    return reconstructed_signal

def generate_functions(x_values):
    functions = {}
    
    # Parabolic Function
    functions['Parabolic'] = np.where((x_values >= -2) & (x_values <= 2), x_values**2, 0)
    
    # Triangular Function
    triangular = np.zeros_like(x_values)
    mask = (x_values >= -2) & (x_values <= 2)
    triangular[mask] = 1 - np.abs(x_values[mask])/2
    functions['Triangular'] = triangular
    
    # Sawtooth Function
    sawtooth = np.zeros_like(x_values)
    sawtooth[mask] = x_values[mask]
    functions['Sawtooth'] = sawtooth
    
    # Rectangular Function
    rectangular = np.zeros_like(x_values)
    rectangular[mask] = 1
    functions['Rectangular'] = rectangular
    
    return functions

def main():
    x_values = np.linspace(-10, 10, 1000)
    frequencies = np.linspace(-5, 5, 1000)
    functions = generate_functions(x_values)
    
    # Create subplots for each function
    for func_name, y_values in functions.items():
        # Create figure with 3 subplots for each function
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle(f"Analysis of {func_name} Function")
        
        # Original function
        ax1.plot(x_values, y_values, label=f"Original {func_name}")
        ax1.set_title("Original Function")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        
        # Fourier Transform
        ft_data = fourier_transform(y_values, frequencies, x_values)
        ax2.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
        ax2.set_title("Frequency Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        
        # Reconstructed function
        reconstructed = inverse_fourier_transform(ft_data, frequencies, x_values)
        ax3.plot(x_values, y_values, label=f"Original {func_name}", color="blue")
        ax3.plot(x_values, reconstructed, label="Reconstructed", color="red", linestyle="--")
        ax3.set_title("Original vs Reconstructed")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
