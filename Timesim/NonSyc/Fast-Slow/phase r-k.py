import numpy as np
import matplotlib.pyplot as plt

# Define parameters
k0 = 10  # Base value for k
dt = 0.01  # Time step
time_end = 3000  # End time for simulation
e = 0.005

# Calculate the time series
time = np.arange(0, time_end, dt)

# Generate k and r values
k_values = k0 + 2 * np.sin(0.01 * time)  # k(t) - oscillatory behavior
r_values = 0.47 + 0.05 * np.sin(0.2 * time) - e * (k0 + 2 * np.sin(0.01 * time))  # r(t)

# Create phase plot of r vs k
plt.figure(figsize=(12, 8))

# Plot r vs k
plt.plot(k_values, r_values, color='purple')

# Adding arrows or lines to indicate direction of movement (using quiver)
plt.quiver(k_values[:-1], r_values[:-1], k_values[1:] - k_values[:-1], r_values[1:] - r_values[:-1],
           scale=5, color='lightgray', alpha=0.5, headlength=2, headaxislength=2)

# Labeling
plt.xlabel('k(t)')
plt.ylabel('r(t)')
plt.title('Phase Space of r(t) vs k(t)')
plt.xlim(8, 12)  # Set limits based on expected values of k(t)
plt.ylim(0, 1)   # Set limits based on expected values of r(t)
plt.grid()
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)  # x-axis line
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)  # y-axis line

# Show plot
plt.tight_layout()
plt.show()