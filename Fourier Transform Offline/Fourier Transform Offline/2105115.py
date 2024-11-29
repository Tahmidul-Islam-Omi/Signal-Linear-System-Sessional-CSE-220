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

def parabolic(x):
    if -3 <= x <= -1 or 1 <= x <= 3:
        return x ** 2
    elif -1 < x < 1:
        return 5 - abs(x)
    else:
        return 0

vec_parabolic = np.vectorize(parabolic)
x_values = np.linspace(-10, 10, 1000) 
y_values = vec_parabolic(x_values)

plt.figure(figsize=(12, 6))    
plt.plot(x_values, y_values)
plt.title("Original")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


frequencies = np.linspace(-5, 5, 1000)
ft_real , ft_img = fourier_transform( y_values, frequencies, x_values)
ft = np.sqrt (ft_real * ft_real + ft_img * ft_img)

# plt.figure(figsize=(12, 6))    
# plt.plot(frequencies , ft)
# plt.title("FT")
# plt.xlabel("F")
# plt.ylabel("magnitude")
# plt.show()

time_dom = np.trapz(y_values * y_values , x_values , dx = x_values[1] - x_values[0])
freq_dom = np.trapz(ft * ft, frequencies, dx=frequencies[1] - frequencies[0])

print(time_dom)
print(freq_dom)