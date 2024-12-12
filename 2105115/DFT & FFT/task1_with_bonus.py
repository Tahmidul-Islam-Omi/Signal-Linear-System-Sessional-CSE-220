import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000



def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    # shift_samples = 3
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B


#implement other functions and logic
#plot the signals
def plot_original_signal(signal, title, color,xlabel="Sample Index",ylabel="Magnitude"):
   
    plt.figure(figsize=(5, 6))
    plt.stem(samples, signal, linefmt=color, markerfmt=color, basefmt=" ")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_magnitude_spectrum(dft_signal, title, color):
    magnitudes = np.abs(dft_signal)
    plt.figure(figsize=(8, 6))
    plt.stem(np.arange(n), magnitudes, linefmt=color, markerfmt=color, basefmt=" ")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.title(title)
    plt.show()

def plot_cross_correlation(cross_corr, title):
    """
    Plot the cross-correlation function, showing the correlation values at different sample lags.

    Parameters:
    cross_corr (array-like): Cross-correlation values.
    title (str): The title of the plot.

    Returns:
    None
    """
    lags = np.arange(-len(cross_corr) // 2, len(cross_corr) // 2)  # Compute lags
    plt.figure(figsize=(8, 6))
    plt.stem(lags, np.roll(cross_corr, len(cross_corr) // 2), linefmt='g', markerfmt='go', basefmt=" ")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.grid()
    plt.show()


#implementation of dft nt using the default
def dft(signal):
    N =len(signal)
    dft_signal = np.zeros(N, dtype=np.complex_)
    
    for k in range(N):
       for n in range(N):
              dft_signal[k] += signal[n] * np.exp(-1j * np.pi * k * 2*n / N)
    return dft_signal
    
def idft(dft_signal): 
    N = len(dft_signal) 
    idft_signal = np.zeros(N, dtype=np.complex_)
    
    for n in range(N):
        for k in range(N):
            idft_signal[n] += dft_signal[k] * np.exp(1j *2* np.pi * k * n / N)
        idft_signal[n] /= N
    return idft_signal  


def cross_correlation(signal_A, signal_B):
    N = len(signal_A)
    dft_signal_A = dft(signal_A)
    dft_signal_B = dft(signal_B)
    dft_conj_signal_A = np.conj(dft_signal_A) # Conjugate of DFT of signal B

    cross_correlation_signal = dft_signal_B * dft_conj_signal_A
    
    cross_correlation_signal = idft(cross_correlation_signal)

    #return only the real part
    return cross_correlation_signal.real

def detect_lag(cross_corr):
    lag = np.argmax(cross_corr)  # Find the index of the max correlation
    if(lag <= len(cross_corr)//2):
        return lag
    else:
        return lag -len(cross_corr)

# Function to apply a low-pass Butterworth filter
def low_pass_filter(signal, cutoff, sampling_rate, order=4):
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Add Gaussian noise
def add_gaussian_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

# Generate noisy signals
signal_A, signal_B = generate_signals()

# Add more noise
signal_A_noisy = add_gaussian_noise(signal_A, noise_level=.2)
signal_B_noisy = add_gaussian_noise(signal_B, noise_level=.2)

# Apply filtering
filtered_signal_A = low_pass_filter(signal_A_noisy, cutoff=20, sampling_rate=sampling_rate)
filtered_signal_B = low_pass_filter(signal_B_noisy, cutoff=20, sampling_rate=sampling_rate)

# Compute cross-correlation with and without filtering
cross_corr_noisy = cross_correlation(signal_A_noisy, signal_B_noisy)
cross_corr_filtered = cross_correlation(filtered_signal_A, filtered_signal_B)

# Detect lag for both cases
lag_noisy = detect_lag(cross_corr_noisy)
lag_filtered = detect_lag(cross_corr_filtered)

# Calculate distances
distance_noisy = np.abs(lag_noisy) / sampling_rate * wave_velocity
distance_filtered = np.abs(lag_filtered) / sampling_rate * wave_velocity

print(f"Sample Lag with Noise: {lag_noisy}, Distance: {distance_noisy} m")
print(f"Sample Lag after Filtering: {lag_filtered}, Distance: {distance_filtered} m")



# Plot results
# plot_original_signal(signal_A_noisy, "Noisy Signal A", "b")
# plot_original_signal(filtered_signal_A, "Filtered Signal A", "g")
# plot_cross_correlation(cross_corr_noisy, "Cross-Correlation (Noisy Signals)")
# plot_cross_correlation(cross_corr_filtered, "Cross-Correlation (Filtered Signals)")
