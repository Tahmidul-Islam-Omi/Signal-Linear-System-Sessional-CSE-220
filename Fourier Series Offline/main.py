import numpy as np
import matplotlib.pyplot as plt

class FourierSeries:
    def __init__(self, func, L, terms=10):
        self.func = func  
        self.L = L     
        self.terms = terms

    def calculate_a0(self, N=1000):
        x = np.linspace(-self.L, self.L , N)
        y = self.func(x)
        a0 = (1/ (2 * self.L)) * np.trapz(y, x)
        return a0

    def calculate_an(self, n, N=1000):
        x = np.linspace(-self.L, self.L , N)
        y = self.func(x)
        an = (2/ (2 * self.L)) * np.trapz(y * np.cos(n * np.pi * x / self.L) , x)
        return an
    def calculate_bn(self, n, N=1000):
        x = np.linspace(-self.L, self.L , N)
        y = self.func(x)
        bn = (2/ (2 * self.L)) * np.trapz(y * np.sin(n * np.pi * x / self.L) , x)
        return bn

    def approximate(self, x):
        
        a0 = self.calculate_a0()
        series = a0 / 2 
        
        for n in range(1, self.terms + 1):
            an = self.calculate_an(n)
            bn = self.calculate_bn(n)
            series += an * np.cos(n * np.pi * x / self.L) + bn * np.sin(n * np.pi * x / self.L)

        return series

    def plot(self):
    
        x = np.linspace(-self.L, self.L, 1000)
        original = self.func(x)
        approximation = self.approximate(x)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, original, label="Original Function", color="blue")
        plt.plot(x, approximation, label=f"Fourier Series Approximation (N={self.terms})", color="red", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Fourier Series Approximation")
        plt.grid(True)
        plt.show()


def target_function(x, function_type="square"):
    
    if function_type == "square":
        return np.sign(np.sin(x))
    
    elif function_type == "sawtooth":
        return 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))
    
    elif function_type == "triangle":
        return 2 * np.abs(2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))) - 1
    
    elif function_type == "sine":
        return np.sin(x)
    
    elif function_type == "cosine":
        return np.cos(x)
    
    else:
        raise ValueError("Invalid function_type. Choose from 'square', 'sawtooth', 'triangle', 'sine', or 'cosine'.")

# Example of using these functions in the FourierSeries class
if __name__ == "__main__":
    L = np.pi  # Half-period for all functions
    terms = 3  # Number of terms in Fourier series

    # Test each type of target function
    for function_type in ["square", "sawtooth", "triangle", "sine", "cosine"]:
        print(f"Plotting Fourier series for {function_type} wave:")
        
        # Define the target function dynamically
        fourier_series = FourierSeries(lambda x: target_function(x, function_type=function_type), L, terms)
        
        # Plot the Fourier series approximation
        fourier_series.plot()
