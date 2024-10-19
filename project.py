import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_data_stream(length=1000, anomaly_chance=0.025):
    time = np.arange(length)
    seasonal_pattern = np.sin(2 * np.pi * time / 100)
    noise = np.random.normal(0, 0.2, size=length)
    data_stream = seasonal_pattern + noise

    for i in range(length):
        if random.random()< anomaly_chance:
            data_stream[i] += random.uniform(2, 5)  # Inject anomalies
        yield data_stream[i]

class AnomalyDetector:
    def __init__(self, window_size=30, threshold=1.7, alpha=0.55):
        self.window_size = window_size
        self.threshold = threshold
        self.alpha = alpha
        self.rolling_window = deque(maxlen=window_size)
        self.ema_value = None

    def process_point(self, data_point):
        if self.ema_value is None:
            self.ema_value = data_point
        
        self.ema_value = self.alpha * data_point + (1 - self.alpha) * self.ema_value
        self.rolling_window.append(data_point)

        if len(self.rolling_window) < self.window_size:
            return False        
        
        print(len(self.rolling_window))

        mean = np.mean(self.rolling_window)
        std = np.std(self.rolling_window)
        
        if std == 0 or np.isinf(std):
            std = 1
            
        z_score = (self.ema_value - mean) / std
        print(f"EMA: {self.ema_value}, Mean: {mean}, Std: {std}, Z-score: {abs(z_score)}, Threshold:{self.threshold}")  # Debugging prints
        return abs(z_score) > self.threshold

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
    new_y = next(data_generator)
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