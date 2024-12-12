import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fft(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = fft(signal[::2])
    odd = fft(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

def ifft(X):
    n = len(X)
    if n <= 1:
        return X
    even = ifft(X[::2])
    odd = ifft(X[1::2])
    T = [np.exp(2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

# Open the encrypted image
image = Image.open("DFT & FFT\\encrypted_image.tiff")

# Convert the image to a NumPy array
encrypted_image = np.array(image)

plt.figure(figsize=(8, 6))

# Encrypted image
plt.subplot(1, 2, 1)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis('off')

# Decryption process
key_row_index = np.argmin(np.sum(encrypted_image, axis=1))  # Find the row with the minimum sum
key_row = encrypted_image[key_row_index]  # Select the key row for decryption
decrypted_image = np.zeros_like(encrypted_image, dtype=np.float64)  # Initialize an empty image for the result

for i in range(encrypted_image.shape[0]):  # Iterate over each row
    encrypted_row_fft = fft(encrypted_image[i])  # FFT of each row
    decrypted_row = ifft(encrypted_row_fft * key_row)  # Apply IFFT with the key row for decryption
    decrypted_image[i] = np.real(decrypted_row)  # Store the real part of the decrypted row

# Normalize and convert to uint8 for image display
decrypted_image = np.clip(decrypted_image, 0, 255).astype(np.uint
