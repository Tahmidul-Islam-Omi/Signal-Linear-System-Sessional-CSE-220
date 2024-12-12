import numpy as np
import matplotlib.pyplot as plt
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000
from scipy.signal import butter, lfilter

#use this function to generate signal_A and signal_B with a random shift
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
    noise_for_signal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_signal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_signal_A 
    noisy_signal_B = signal_A + noise_for_signal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    # shift_samples = 5
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B, shift_samples

def filtering(signal , threshold_ratio=0.1):
    dft_signal = dft(signal)
    magnitudes = np.abs(dft_signal)
    threshold = threshold_ratio * np.max(magnitudes)
    
    for i in range(len(dft_signal)):
        if magnitudes[i] < threshold:
            dft_signal[i] = 0
            
    filtered_signal = idft(dft_signal)
    
    return filtered_signal.real

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exponent = np.exp(-2j * np.pi * k * n / N)
    return np.dot(exponent, x)

def idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exponent = np.exp(2j * np.pi * k * n / N)
    return np.dot(exponent, X) / N

def cross_correlation(signal_A, signal_B):
    
    X_A = dft(signal_A)
    X_B = dft(signal_B)
    
    corr = X_A * np.conj(X_B) 
    corr_time_domain = idft(corr)
    
    return np.real(corr_time_domain)

def detect_lag(cross_corr):
    lag = np.argmax(cross_corr) 
    if(lag <= len(cross_corr)//2):
        return lag
    else:
        return lag - len(cross_corr)


def estimate_distance(lag, sampling_rate, wave_velocity):
    time_lag = lag / sampling_rate  
    distance = np.abs(time_lag * wave_velocity) 
    return distance

#implement other functions and logic
def main():
    signal_A, signal_B, true_lag = generate_signals()
    
    plt.figure(figsize=(10, 6))
    plt.stem(samples, signal_A)
    plt.title("Signal A (Station A)")
    plt.xlabel("Sample Index")
    plt.show()
    
    dft_A = dft(signal_A)
    mag_A = np.abs(dft_A)
    
    plt.figure(figsize=(10, 6))
    plt.stem(samples, mag_A)
    plt.title("Frequency Spectrum of Signal A")
    plt.ylabel("Magnitude")
    plt.show()
    
    # ..............................
    
    plt.figure(figsize=(10, 6))
    plt.stem(samples, signal_B)
    plt.title("Signal B (Station B)")
    plt.xlabel("Sample Index")
    plt.show()
    
    dft_B = dft(signal_B)
    mag_B = np.abs(dft_B)
    
    plt.figure(figsize=(10, 6))
    plt.stem(samples, mag_B)
    plt.title("Frequency Spectrum of Signal B")
    plt.ylabel("Magnitude")
    plt.show()
    
    cross_corr = cross_correlation(signal_A, signal_B)
    

    plt.figure(figsize=(10, 6))
    plt.stem( np.arange(-len(cross_corr) // 2, len(cross_corr) // 2), np.roll(cross_corr, len(cross_corr) // 2))
    plt.title("Cross-Correlation between Signal A and Signal B")
    plt.ylabel("Correlation Value")
    plt.show()
    
    detected_lag = detect_lag(cross_corr)
    print(f"True Lag: {true_lag}, Detected Lag: {detected_lag}")


    distance = estimate_distance(detected_lag, sampling_rate, wave_velocity)
    print(f"Estimated Distance between Stations: {distance} meters")
def main_with_filtering(fr):
    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    frequency = 5
    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_signal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_signal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_signal_A 
    noisy_signal_B = signal_A + noise_for_signal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    # shift_samples = 5
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples) 
    
    filtered_signal_A = filtering(signal_A)
    filtered_signal_B = filtering(signal_B)   
    
    
    
# main()    
# main_with_filtering()