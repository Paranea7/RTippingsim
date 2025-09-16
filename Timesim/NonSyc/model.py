import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit

# Define parameters
alpha = 1.0
beta = 1.0
dt = 0.01
time_end = 100.0
num_steps = int(time_end / dt)

# Sinusoidal frequency
wk = 0.02
t_values = np.linspace(0.0, time_end, num_steps)  # Time array
X_initial_conditions = np.array([0.1, 5.0])  # Initial conditions
mean_k = np.mean(10.0 + np.sin(wk * t_values))  # Precompute mean k value

@jit(nopython=True)
def rk4_vectorized(x0, r_value, k_values, dt, num_steps, alpha, beta):
    x = x0  # Assigning directly
    for k in k_values:  # Iterate through k values
        if k <= 0:
            continue  # skip non-positive k values
        for _ in range(num_steps):
            k1 = dt * (r_value * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
            k2 = dt * (r_value * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                       (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r_value * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                       (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r_value * (x + k3) * (1 - (x + k3) / k) -
                       (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

@jit(nopython=True)
def compute_single_steady_state(x0, r_value, k_values, dt, num_steps, alpha, beta):
    # Directly modifying x0 in the loop
    x = x0
    for k in k_values:
        if k <= 0:
            continue
        for _ in range(num_steps):
            k1 = dt * (r_value * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
            k2 = dt * (r_value * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                       (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r_value * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                       (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r_value * (x + k3) * (1 - (x + k3) / k) -
                       (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

def compute_steady_state(r_value):
    results = []
    k_values = 10.0 + np.sin(wk * t_values)  # Compute k_values
    for x0 in X_initial_conditions:
        final_x = compute_single_steady_state(x0, r_value, k_values, dt, num_steps, alpha, beta)
        results.append(final_x)
    return results

if __name__ == '__main__':
    # Generate r values, from 0 to 1 with step of 0.02
    r_values = np.linspace(0.0, 1.0, 50)

    # Use multiprocessing to compute steady states
    with Pool() as pool:
        steady_states = pool.map(compute_steady_state, r_values)

    # Process steady state results
    x_stable = []
    roots_np = []

    for r_value, steady_x_list in zip(r_values, steady_states):
        for steady_x in steady_x_list:
            x_stable.append((r_value, steady_x))

        # Calculate equilibrium points
        roots = np.roots(
            [-r_value, mean_k * r_value,
             -(mean_k + r_value),
             mean_k * r_value]
        )
        real_roots = roots[np.isreal(roots)].real
        for root in real_roots:
            roots_np.append((r_value, root))

    # Separate results into r and x lists
    r_plot, x_plot = zip(*[(r_value, steady_x) for r_value, steady_x in x_stable])

    # Group by unique r values
    r_unique = sorted(set(r_plot))
    x_bistable_1 = []
    x_bistable_2 = []

    for r in r_unique:
        x_vals = [x_plot[i] for i in range(len(x_plot)) if r_plot[i] == r]
        if len(x_vals) > 0:
            x_bistable_1.append(max(x_vals))
            x_bistable_2.append(min(x_vals))

    # Separate roots into r and x arrays
    r_points_np, x_points_np = zip(*roots_np) if roots_np else ([], [])

    # Plot bistable curve
    plt.figure(figsize=(12, 6))
    plt.scatter(r_points_np, x_points_np, s=10, c='y', label='Using np.roots', zorder=2)
    plt.plot(r_unique, x_bistable_1, marker='o', label='Stable State 1', color='blue', zorder=1)
    plt.plot(r_unique, x_bistable_2, marker='x', label='Stable State 2', color='red', zorder=1)
    plt.xlabel('r')
    plt.ylabel('Steady state x')
    plt.title('Bistability in the System with k = 10 + sin(wk * t)')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 11, 1))
    plt.legend()
    plt.grid()
    plt.show()