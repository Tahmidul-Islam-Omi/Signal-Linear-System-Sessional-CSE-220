import numpy as np
import matplotlib.pyplot as plt

# Time domain settings
sampling_rate = 1000  # Samples per second
duration = 1          # Signal duration in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)  # Time vector

# Define the signal: 3*sin(2π5t) + 2*sin(2π15t) + 1*sin(2π30t)
signal = 3 * np.sin(2 * np.pi * 5 * t) + 2 * np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 30 * t)

# Perform Fourier Transform
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(fft_result), 1 / sampling_rate)  # Frequency bins
magnitude = np.abs(fft_result)

# Plot time-domain signal and frequency-domain representation
plt.figure(figsize=(12, 6))

# Time domain plot
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Time-Domain Signal')
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# Frequency domain plot (only positive frequencies)
plt.subplot(2, 1, 2)
plt.stem(fft_freq[:len(fft_freq)//2], magnitude[:len(magnitude)//2], basefmt=" ")
plt.title('Frequency Domain (Fourier Transform)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()

plt.show()
