import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 定义参数
k0 = 10
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)

# RK4 方法
def rk4_vectorized(x0s, r_values, k_values, dt):
    x_values = np.zeros_like(x0s)  # 初始化存储状态的数组
    x_values[:] = x0s  # 设置初始条件

    for t in range(num_steps):
        x = x_values.copy()  # 保存上一个时间步骤的 x 值

        r = r_values[t]
        k = k_values[t]

        k1 = dt * (r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
        k2 = dt * (r * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                   (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
        k3 = dt * (r * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                   (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
        k4 = dt * (r * (x + k3) * (1 - (x + k3) / k) -
                   (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))

        x_values = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values

# 读取 all_w_zero.txt 中的 w 值
def read_w_values(filename):
    with open(filename, 'r') as file:
        w_values = [float(line.strip()) for line in file.readlines() if line.strip()]
    return w_values

# 对于每个 w 值进行计算
def compute_dynamics(w):
    r0 = 0.47
    time_points = np.arange(0, time_end, dt)
    r_values = r0 + 0.02 * np.sin(w * time_points)  # 动态 r 值
    k_values = k0 + 2 * np.sin(w * time_points)     # 动态 k 值

    # 生成初始值
    x0_values = np.linspace(0, 7, 71)  # 生成初始值
    all_x_values = []  # 存储每个初始条件的演化

    # 计算每个初始值的演化
    for x0 in x0_values:
        evolution = rk4_vectorized(x0s=np.array([x0]), r_values=r_values, k_values=k_values, dt=dt)
        all_x_values.append(evolution)

    return w, time_points, np.array(all_x_values)

# 绘制结果
def plot_dynamics(results):
    for w, time_points, all_x_values in results:
        plt.figure(figsize=(12, 6))
        x0_values = np.linspace(0, 7, 71)  # 生成初始值

        for i in range(len(x0_values)):
            color = 'red' if x0_values[i] > 2.32 else 'blue'
            plt.plot(time_points, all_x_values[i], color=color, alpha=0.5)

        plt.xlabel('Time')
        plt.ylabel('x')
        plt.title(f'Dynamics of x over time for w={w}, r0={0.47} and k0={k0}')
        plt.axhline(y=2.32, color='grey', linestyle='--', label='x = 2.32')  # 添加参考线
        plt.grid()
        plt.legend(["x > 2.32", "x ≤ 2.32"], loc='upper left')
        plt.tight_layout()  # 调整图形以适应标签
        plt.show()

# 主程序
if __name__ == "__main__":
    all_w_zero = read_w_values('tipping rates/all_w_zero.txt')

    # 使用多进程处理
    with Pool() as pool:
        results = pool.map(compute_dynamics, all_w_zero)

    # 绘制图形
    plot_dynamics(results)