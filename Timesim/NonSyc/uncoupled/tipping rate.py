import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit  # 导入 numba 库

# 定义参数
k0 = 10  # k的基础值
dt = 0.01  # 时间步长
time_end = 100  # 仿真时间
num_steps = int(time_end / dt)

# 模型参数
alpha = 1.0  # 参数 alpha
beta = 1.0  # 参数 beta

# 定义初值和终值的阈值
initial_threshold = 2.32  # 初值的阈值
final_threshold = 3  # 终值的阈值

# 固定的 wk 值
wk = 0.01


@njit  # 加速的 RK4 方法
def rk4_optimized(x0s, wr, dt, num_steps):
    num_initial_conditions = len(x0s)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 存储每个初始条件的 x 值
    x_values[:, 0] = x0s  # 设置初始值

    for i in range(1, num_steps):
        t = i * dt  # 当前时间
        r = 0.47 + 0.05 * np.sin(wr * t)  # 使用 wr 计算 r 的值
        k = k0 + 2 * np.sin(wk * t)

        for j in range(num_initial_conditions):
            x = x_values[j, i - 1]  # 获取前一步的 x 值

            k1 = dt * (r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
            k2 = dt * (r * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                       (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                       (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r * (x + k3) * (1 - (x + k3) / k) -
                       (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))

            x_values[j, i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values


# 计算 tipping rate 的函数
def calculate_tipping_rate(wr):
    initial_conditions = np.linspace(0, 4.73, 2000)  # 初始条件范围
    initial_conditions = initial_conditions[initial_conditions < initial_threshold]  # 过滤初值符合条件的初始条件

    if len(initial_conditions) == 0:  # 如果没有符合初值条件的初始条件
        return 0

    x_values = rk4_optimized(initial_conditions, wr, dt, num_steps)  # 使用加速的RK4求解 x 的值
    final_x = x_values[:, -1]  # 取每个初始条件最终的 x 值

    n = np.sum(final_x > final_threshold)  # 计算满足终值条件的计数
    tipping_rate = n / len(initial_conditions)  # 计算 tipping rate

    return tipping_rate


# wr 的范围
wr_values = np.linspace(0.1, 1.10, 1110)  # wr的范围

# 使用 Pool 进行并行计算
if __name__ == "__main__":
    with Pool() as pool:
        tipping_rates = pool.map(calculate_tipping_rate, wr_values)

    # 绘制散点图并连线
    plt.figure(figsize=(12, 6))
    plt.plot(wr_values, tipping_rates, marker='o', linestyle='-', color='blue')
    plt.xlabel('$w_r$')
    plt.ylabel('Tipping Rate')
    plt.title(f'Tipping Rate vs. $w_r$ with Fixed $w_k = {wk}$ and Initial Threshold = {initial_threshold} and Final Threshold = {final_threshold}')
    plt.grid()
    plt.tight_layout()
    plt.show()