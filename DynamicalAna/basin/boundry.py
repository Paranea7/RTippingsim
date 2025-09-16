import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define parameters
r = 0.42   # Growth rate
k = 12   # Carrying capacity
alpha = 1.0
beta = 1.0

# Define the model equation
def model(x):
    return r * x * (1 - x / k) - beta * x / (alpha**2 + x**2)

# Find fixed points over a range
x_values = np.linspace(0, k, 10000)  # Set the range for x
dxdt_values = model(x_values)

# Find the roots where dx/dt = 0
root_indexes = np.where(np.diff(np.sign(dxdt_values)))[0]
fixed_points = x_values[root_indexes]

# Print the locations of fixed points
print(f"Fixed points: {fixed_points}")

# Plot the phase space and attractive domains
plt.figure(figsize=(10, 6))
plt.plot(x_values, dxdt_values, label='dx/dt', color='blue')
plt.axhline(0, color='gray', lw=0.5, linestyle='--')

# Mark all fixed points
for fp in fixed_points:
    plt.plot(fp, model(fp), 'ro')  # Mark fixed points with red dots

# Mark attractive domains
for i in range(len(fixed_points) - 1):
    plt.fill_between(x_values, dxdt_values, where=(x_values > fixed_points[i]) & (x_values < fixed_points[i + 1]),
                     color='lightgreen', alpha=0.5)

plt.title('Fixed Points and Their Attractive Domains')
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.legend()
plt.grid()
plt.show()