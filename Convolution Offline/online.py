import numpy as np
import matplotlib.pyplot as plt

class DiscreteSignal:
    def __init__(self, INF):
        self.INF = INF        
        self.values =  np.zeros(self.INF)
        
    def set_value_at_time(self, time, value):
        self.values[time] = value
    
    def shift_signal(self, shift):
        shifted_signal  = DiscreteSignal(self.INF)
        
        if shift < 0:
            shifted_signal.values[-shift :] = self.values[0 : len(self.values) + shift]
        elif shift > 0:
            shifted_signal.values[0 : len(self.values) - shift ] = self.values[shift:]
        else:
            shifted_signal.values = self.values.copy();    
        return shifted_signal 
    
    def add(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Two Signals must have the same length to be added.")
        added_signal = DiscreteSignal(self.INF)
        added_signal.values = self.values + other.values
        return added_signal
    
    def multiply(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Two Signals must have the same length to be multiplied.")
        multiplied_signal = DiscreteSignal(self.INF)
        multiplied_signal.values = self.values * other.values
        return multiplied_signal  
    
    def multiply_const_factor(self, scaler):
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values * scaler
        return new_signal 

class LTI_Discrete:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response
        self.INF = impulse_response.INF
        self.values = impulse_response.values
        
    def linear_combination_of_impulses(self, input_signal):
        for i in range(len(input_signal.values)):
            signal = DiscreteSignal(self.INF)
            signal.values[i] = input_signal.values[i]
            signal.plot('n(Time Index)' , 'x[n]' , f"$[n-({i-self.INF})]x[{i-self.INF}]")
        input_signal.plot('n(Time Index)' , 'x[n]' , 'Sum')    
            
    def output(self, input_signal, end_Idx):
        
        for i in range(0, end_Idx):
            shifted_impulse = DiscreteSignal(self.INF)
            shifted_impulse = self.impulse_response.shift_signal(-i)
            
            signal = DiscreteSignal(self.INF)
            signal = shifted_impulse.multiply(input_signal)
            
            output = 0
            for j in range(len(signal.values)):
                output += signal.values[j]
            print(f"{output:.2f}" , end=" ")
        

if __name__ == '__main__':
    # Stock Market Prices as a Python List
    price_list = list(map(int, input("Stock Prices: ").split()))
    n = int(input("Window size: "))        

    # price_list = [1, 2, 3, 4, 5, 6, 7, 8]
    # n = 4

    # Please determine uma and wma.

    # Unweighted Moving Averages as a Python list
    num = 1 / n
    uma = []
    
    for i in range(0, n):
        uma.append(num)

    # Weighted Moving Averages as a Python list
    sum = 0
    for i in range(1, n+1):
        sum += i
        
    wma = []
    for i in range(1,n+1):
        wma.append(i/sum)
        
    INF = len(price_list)
    input_signal = DiscreteSignal(INF)
    
    
    for i in range(0, len(price_list)):
        input_signal.set_value_at_time(i , price_list[i])
    
    impulse_signal = DiscreteSignal(INF)    
    for i in range(0, len(uma)):
        impulse_signal.set_value_at_time(i , uma[i])    
    
    end_Idx = len(price_list) - n + 1
    obj = LTI_Discrete(impulse_signal)
    print("Unweighted Moving Averages: " , end=" ")
    obj.output(input_signal , end_Idx)
    
    print()
    
    impulse_signal2 = DiscreteSignal(INF)
    for i in range(0, len(wma)):
        impulse_signal2.set_value_at_time(i , wma[i])   
    
    obj2 = LTI_Discrete(impulse_signal2)
    
    print("Weighted Moving Averages: ", end=" ")
    obj2.output(input_signal , end_Idx)   
