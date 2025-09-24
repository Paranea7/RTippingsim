import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 定义参数
V_c = 10.0
D = 0.001
dt = 0.01  # 步长
time_end = 200  # 仿真结束时间
num_steps = int(time_end / dt)
c = 1.0  # 常数 c
nx = ny = 50


# 使用 NumPy 中的滤波器来实现拉普拉斯算子
def laplacian(V):
    return (
            np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) -
            4 * V
    )


# 向量化的 RK4 方法
def rk4(V0, r_func, dt, num_steps):
    V = V0
    V_values = np.zeros((num_steps, V0.shape[0], V0.shape[1]))  # 存储每个时间步的 V 值
    for i in range(num_steps):
        r_value = r_func(i * dt)  # 计算当前时间的 r(t)

        # 根据新的动态方程计算 k1, k2, k3, k4
        k1 = dt * (r_value * V * (1 - V / V_c) - (c * V ** 2) / (V ** 2 + 1) + D * laplacian(V))
        k2 = dt * (r_value * (V + 0.5 * k1) * (1 - (V + 0.5 * k1) / V_c) - (c * (V + 0.5 * k1) ** 2) / (
                (V + 0.5 * k1) ** 2 + 1) + D * laplacian(V + 0.5 * k1))
        k3 = dt * (r_value * (V + 0.5 * k2) * (1 - (V + 0.5 * k2) / V_c) - (c * (V + 0.5 * k2) ** 2) / (
                (V + 0.5 * k2) ** 2 + 1) + D * laplacian(V + 0.5 * k2))
        k4 = dt * (r_value * (V + k3) * (1 - (V + k3) / V_c) - (c * (V + k3) ** 2) / (
                (V + k3) ** 2 + 1) + D * laplacian(V + k3))

        V += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        V_values[i] = V  # 记录每个时间步的 V 值
    return V_values  # 返回所有时间步的 V 值


# 固定 r 和生成初始条件
def r_func(t):
    return 0.47 + 0.05 * np.sin(0.1 * t)


# 定义并行计算的函数
def compute_for_initial_condition(V0):
    initial_state = np.random.normal(loc=V0, scale=0.1, size=(nx, ny))  # 为整个空间生成随机初值
    V_values = rk4(initial_state, r_func, dt, num_steps)

    # 使用 NumPy 的均值计算
    mean_values = np.mean(V_values, axis=(1, 2))  # 计算每个时间步的均值
    return mean_values


# 设置初始值范围
initial_conditions = np.linspace(1, 9, 100)  # 生成160个均匀分布的初始值

# 使用 multiprocessing.Pool 进行并行计算
if __name__ == '__main__':
    with Pool() as pool:
        results = pool.map(compute_for_initial_condition, initial_conditions)

    # 绘制结果，每个初始条件对应一条曲线
    plt.figure(figsize=(12, 6))
    time_vector = np.arange(0, time_end, dt)

    for mean_values in results:
        plt.plot(time_vector, mean_values, color='blue', alpha=0.1)  # 每条曲线透明度稍低

    # 计算并绘制均值曲线
    mean_of_means = np.mean(results, axis=0)  # 结合所有初始条件的均值
    plt.plot(time_vector, mean_of_means, color='red', label='Mean of Means', linewidth=2)  # 全局均值曲线

    # 添加标签和标题
    plt.xlabel('Time')
    plt.ylabel('Mean V')
    plt.title('Mean V over time for varying initial conditions')
    plt.legend()
    plt.grid()
    plt.tight_layout()  # 调整图形以适应标签
    plt.show()