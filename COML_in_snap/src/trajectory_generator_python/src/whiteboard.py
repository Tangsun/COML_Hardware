import numpy as np

start_point = np.array([0, 0, 1])
end_point = np.array([1, 1, 1])

# Define total time duration for the trajectory
total_time = 1.0  # seconds

# Calculate constant velocity
velocity = (end_point - start_point) / total_time

# Create time array
time_array = np.linspace(0, total_time, num=3000)  # 3000 timesteps

# Calculate position, velocity, and acceleration matrices
r = np.outer(time_array, velocity) + np.tile(start_point, (len(time_array), 1))
dr = np.tile(velocity, (len(time_array), 1))
ddr = np.zeros_like(r)