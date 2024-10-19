import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_data_stream(length=1000, anomaly_chance=0.025):
    if length <=0:
        raise  ValueError("Length must be greater than 0")
    if  anomaly_chance < 0 or anomaly_chance > 1:
        raise ValueError("Anomaly chance must be between 0 and 1")

    time = np.arange(length)# Map each index in an array can be thought of as an instance in time i.e each index is nth time step
    seasonal_pattern = np.sin(2 * np.pi * time / 100) #the use of sin allows us to create a cyclical pattern, allowing for seasonal trends, every 100 times steps, the pattern will repeat
    noise = np.random.normal(0, 0.2, size=length)  #this is the random noise that we will add to the seasonal pattern to create the data stream

    data_stream = seasonal_pattern + noise

    for i in range(length):
        if random.random()< anomaly_chance:  #if the random number is less than the anomaly chance, we will introduce an anomaly the random number is between 0 and 1
            data_stream[i] += random.uniform(2, 5)  # Inject anomalies 
        yield data_stream[i]

class AnomalyDetector:
    def __init__(self, window_size=30, threshold=3, alpha=1):
        self.window_size = window_size #This window size is the number of data points that we will use to calculate the mean and standard deviation
        self.threshold = threshold  #This is the number of standard deviations that we will use to determine if a data point is an anomaly i.e if the data point is more than 3 standard deviations away from the mean, it is an anomaly
        self.alpha = alpha   #This is the smoothing factor that we will use to calculate the exponentially weighted moving average, the higher the alpha the more weight we will give to the most recent data points
        self.rolling_window = deque(maxlen=window_size)
        self.ema_value = None
        #Error Checks to see if the right values were passed
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer")
        if threshold <= 0:
            raise ValueError("Threshold must be a positive number")
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        ####
    def process_point(self, data_point):
        #Error checks to see if data point is a valid data point
        if data_point is None:
            raise ValueError("Data point cannot be type : None")
        if not data_point:
            raise ValueError("No data point found")
        if not isinstance(data_point, float):
            raise TypeError (f"Data point must be a float received {type(data_point)}")
        ####    
        if self.ema_value is None:
            self.ema_value = data_point 
        
        self.ema_value = self.alpha * data_point + (1 - self.alpha) * self.ema_value #Calculation to determine the ema value of the current data point
        self.rolling_window.append(data_point)

        if len(self.rolling_window) < self.window_size: #We wait to get enough datapoints to populate our window to give an accurate representation of our mean and standard deviation values
            return False        
        

        mean = np.mean(self.rolling_window)
        std = np.std(self.rolling_window)
        
        if std == 0 or np.isinf(std):  #Avoid division by 1 or infinity
            std = 1
            
        z_score = (self.ema_value - mean) / std  #Calculate the z score of the current data point, this is the number of standard deviations that the data point is away from the mean

        # print(f"EMA: {self.ema_value}, Mean: {mean}, Std: {std}, Z-score: {abs(z_score)}, Threshold:{self.threshold}")  # Debugging prints
        return abs(z_score) > self.threshold   #Return True if the z score is greater than the threshold, indicating that the data point is an anomaly

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6)) 
line, = ax.plot([], [], lw=2, label='Data Stream')
anomaly_scatter, = ax.plot([], [], 'ro', markersize=8, label='Anomalies')
ax.set_xlim(0, 200)
ax.set_ylim(-7, 7)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Real-time Anomaly Detection')
ax.legend()

# Initialize data
x_data, y_data = [], []
anomaly_x, anomaly_y = [], []

# Create instances of generator and detector
data_generator = generate_data_stream()
detector = AnomalyDetector()

def update(frame):  
    global x_data, y_data, anomaly_x, anomaly_y
    
    # Get new data point
    try:
        new_y = next(data_generator)
    except StopIteration:
        print("Data stream ended")
        return line, anomaly_scatter
    x_data.append(frame)
    y_data.append(new_y)
    
    # Detect anomaly
    is_anomaly = detector.process_point(new_y)
    if is_anomaly:
        anomaly_x.append(frame)
        anomaly_y.append(new_y)
        print(f"Anomaly detected at point {frame}: {new_y}")
    
    # Update plot data
    line.set_data(x_data, y_data)
    anomaly_scatter.set_data(anomaly_x, anomaly_y)
    
    # Adjust x-axis limit for scrolling effect
    if frame > 200:
        ax.set_xlim(frame - 200, frame)
    
    return line, anomaly_scatter

# Create animation
ani = FuncAnimation(fig, update, frames=range(1000), blit=True, interval=50)

plt.show()