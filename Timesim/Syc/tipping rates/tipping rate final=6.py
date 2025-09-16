import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 定义参数
k0 = 10  # k的基础值
dt = 0.01  # 时间步长
time_end = 100  # 仿真时间
num_steps = int(time_end / dt)

# 模型参数
alpha = 1.0  # 参数 alpha
beta = 1.0   # 参数 beta

# 定义初值和终值的阈值
initial_threshold = 2.32  # 初值的阈值
final_threshold = 3  # 终值的阈值

# 向量化的 Runge-Kutta 方法
def rk4_vectorized(x0, w, dt, num_steps, amplitude_k, amplitude_r):
    x_values = np.zeros((len(x0), num_steps))
    x_values[:, 0] = x0

    for i in range(1, num_steps):
        t = i * dt
        r = 0.47 + amplitude_r * np.sin(w * t)  # 动态 r 值带振幅
        k = k0 + amplitude_k * np.sin(w * t)  # 动态 k 值带振幅

        # 计算 k1, k2, k3, k4
        k1 = dt * (r * x_values[:, i-1] * (1 - x_values[:, i-1] / k) -
                    (beta * x_values[:, i-1] ** 2) / (alpha ** 2 + x_values[:, i-1] ** 2))
        k2 = dt * (r * (x_values[:, i-1] + 0.5 * k1) * (1 - (x_values[:, i-1] + 0.5 * k1) / k) -
                    (beta * (x_values[:, i-1] + 0.5 * k1) ** 2) / (alpha ** 2 + (x_values[:, i-1] + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_values[:, i-1] + 0.5 * k2) * (1 - (x_values[:, i-1] + 0.5 * k2) / k) -
                    (beta * (x_values[:, i-1] + 0.5 * k2) ** 2) / (alpha ** 2 + (x_values[:, i-1] + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_values[:, i-1] + k3) * (1 - (x_values[:, i-1] + k3) / k) -
                    (beta * (x_values[:, i-1] + k3) ** 2) / (alpha ** 2 + (x_values[:, i-1] + k3) ** 2))

        x_values[:, i] = x_values[:, i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values[:, -1]  # 返回最后一个时间步的 x 值作为数组

# 计算 tipping rate 的函数
def calculate_tipping_rate(params):
    w, amplitude_k, amplitude_r = params
    initial_conditions = np.linspace(0, 4, 800)
    below_initial_threshold = initial_conditions[initial_conditions < initial_threshold]

    # 数量 N
    N = len(below_initial_threshold)

    # 计算所有 x0 的最终 x 值
    final_x_values = rk4_vectorized(below_initial_threshold, w, dt, num_steps, amplitude_k, amplitude_r)

    # 计算 n
    n = np.sum(final_x_values > final_threshold)

    # 计算 tipping rate
    tipping_rate = n / N if N > 0 else 0
    return tipping_rate

# w 的范围
w_values = np.linspace(0.01, 0.66, 555)  # w的范围

# 使用 Pool 进行并行计算
if __name__ == "__main__":
    amplitudes_k = [0.5, 1.0, 2.0]  # k的振幅列表
    amplitudes_r = [0.02, 0.03, 0.05]  # r的振幅列表
    results = {f'k={a_k}, r={a_r}': [] for a_k in amplitudes_k for a_r in amplitudes_r}

    # 对于每种 k 和 r 的组合，计算 tipping rates
    for amplitude_k in amplitudes_k:
        for amplitude_r in amplitudes_r:
            params = [(w, amplitude_k, amplitude_r) for w in w_values]

            with Pool() as pool:
                tipping_rates = pool.map(calculate_tipping_rate, params)

            results[f'k={amplitude_k}, r={amplitude_r}'] = tipping_rates

    # 绘制散点图并连线
    plt.figure(figsize=(12, 8))
    for key, tipping_rates in results.items():
        plt.plot(w_values, tipping_rates, marker='o', linestyle='-', label=key)

    plt.xlabel('$w$')
    plt.ylabel('Tipping Rate')
    plt.title('Tipping Rate vs. $w$ for Different Amplitudes of $k$ and $r$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()