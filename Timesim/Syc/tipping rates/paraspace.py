import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from numba import njit

# 定义参数
k0 = 10  # k的基础值
dt = 0.01  # 时间步长
time_end = 500  # 仿真时间
num_steps = int(time_end / dt)

# 模型参数
alpha = 1.0  # 参数 alpha
beta = 1.0  # 参数 beta

# 定义初值和终值的阈值
initial_threshold = 2.36  # 初值的阈值
final_threshold = 2.4  # 终值的阈值

@njit
def rk4_optimized(x0s, w, ampr, ampk, dt, num_steps):
    num_initial_conditions = len(x0s)
    x_values = np.zeros((num_initial_conditions, num_steps))
    x_values[:, 0] = x0s

    for i in range(1, num_steps):
        t = i * dt
        r = 0.47 + ampr * np.sin(w * t)
        k = k0 + ampk * np.sin(w * t)

        for j in range(num_initial_conditions):
            x = x_values[j, i - 1]

            k1 = dt * (r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
            k2 = dt * (r * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                       (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                       (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r * (x + k3) * (1 - (x + k3) / k) -
                       (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))

            x_values[j, i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values

def calculate_tipping_rate(params):
    w, ampr, ampk = params
    initial_conditions = np.linspace(0, 4.73, 2000)
    initial_conditions = initial_conditions[initial_conditions < initial_threshold]

    if len(initial_conditions) == 0:
        return 0

    x_values = rk4_optimized(initial_conditions, w, ampr, ampk, dt, num_steps)
    final_x = x_values[:, -1]

    n = np.sum(final_x > final_threshold)
    tipping_rate = n / len(initial_conditions)

    return tipping_rate

# 组合参数
w_values = np.linspace(0.01, 1.10, 111)  # w的范围
ampr_values = np.linspace(0.02, 0.05, 100)  # 更细的 r 振幅取值范围
ampk_values = np.linspace(0.5, 2.0, 100)  # 更细的 k 振幅取值范围

if __name__ == "__main__":
    # 创建参数组合
    param_combinations = [(w, ampr, ampk) for w in w_values for ampr in ampr_values for ampk in ampk_values]

    # 并行计算所有组合的 tipping rate
    with Pool() as pool:
        tipping_rates = pool.map(calculate_tipping_rate, param_combinations)

    # 将结果整理为适合绘图的格式
    tipping_rates = np.array(tipping_rates).reshape(len(w_values), len(ampr_values), len(ampk_values))

    # 绘制三维参数空间图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    W, Ampr, Ampk = np.meshgrid(w_values, ampr_values, ampk_values, indexing='ij')
    TippingRates = tipping_rates.flatten()

    # 绘制三维散点图
    scatter = ax.scatter(W.flatten(), Ampr.flatten(), Ampk.flatten(), c=TippingRates, cmap='viridis', marker='o')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Tipping Rate')

    # 设置轴标签
    ax.set_xlabel('$w$')
    ax.set_ylabel('Amplitude of r ($ampr$)')
    ax.set_zlabel('Amplitude of k ($ampk$)')
    ax.set_title('Tipping Rate in 3D Parameter Space')

    plt.show()