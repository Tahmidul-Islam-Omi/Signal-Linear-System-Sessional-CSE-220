import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Your custom FFT function
def fft(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = fft(signal[::2])
    odd = fft(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

# Your custom IFFT function
def ifft(X):
    n = len(X)
    if n <= 1:
        return X
    even = ifft(X[::2])
    odd = ifft(X[1::2])
    T = [np.exp(2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

# Load the encrypted image
image = Image.open("DFT & FFT/encrypted_image.tiff")
# Convert the image to a NumPy array
encrypted_image = np.array(image)

key_row_index = np.argmin(np.sum(encrypted_image, axis=1))
key_row = encrypted_image[key_row_index]
key_row_fft = np.array(fft(key_row))

decrypted_image = np.zeros_like(encrypted_image, dtype=np.float64)

for i in range(encrypted_image.shape[0]):
    if i == key_row_index:
        decrypted_image[i] = key_row
    else:
        encrypted_fft = fft(encrypted_image[i])
        decrypted_fft = encrypted_fft / key_row_fft
        decrypted_image[i] = np.array(ifft(decrypted_fft.tolist())).real


# Encrypted image
plt.subplot(1, 2, 1)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis('off')

# Decrypted image
plt.subplot(1, 2, 2)
plt.imshow(decrypted_image, cmap='gray')
plt.title("Decrypted Image")
plt.axis('off')

plt.show()
