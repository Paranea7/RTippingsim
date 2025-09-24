import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 参数设置
r = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
p = 4.0
E0_values = np.linspace(4, 6, 21)  # 从1到10，生成21个E0值
h_E = 1.5  # 设定的常数
h_v = 0.2  # 设定的常数

# 初始化网格和随机初始V
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化
def rk4_step(V, E0):
    E = E0 * h_v / (h_v + V)  # 根据新的公式计算E
    k1 = (r * V * (1 - (h_E**p + E**p) / h_E**p * V) + D * laplacian(V))
    k2 = (r * (V + 0.5 * dt * k1) * (1 - (h_E**p + (E + 0.5 * dt * k1)**p) / h_E**p * (V + 0.5 * dt * k1)) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r * (V + 0.5 * dt * k2) * (1 - (h_E**p + (E + 0.5 * dt * k2)**p) / h_E**p * (V + 0.5 * dt * k2)) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r * (V + dt * k3) * (1 - (h_E**p + (E + dt * k3)**p) / h_E**p * (V + dt * k3)) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next

# 计算每个E0值下的结果
def compute_for_E0(E0):
    V = np.random.uniform(10, 10.1, (nx, ny))  # 每次重新初始化V，生成在10到10.1之间的均匀随机数
    for t in range(15000):  # 迭代15000次，时间步数
        V = rk4_step(V, E0)
    return V  # 返回最终的V

# 使用多进程计算不同E0值得到的V
if __name__ == '__main__':
    with Pool() as pool:  # 使用全部可用的CPU核心
        results = pool.map(compute_for_E0, E0_values)

    # 计算均值
    mean_values = [np.mean(result) for result in results]  # 原始均值计算
    # 添加高斯白噪声
    noise_level = 1.0  # 噪声强度
    noisy_results = [result + np.random.normal(0, noise_level, result.shape) for result in results]
    mean_noisy_values = [np.mean(noisy_result) for noisy_result in noisy_results]  # 噪声版本均值计算

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(E0_values, mean_values, marker='o', label='Original Mean V')  # 原始均值曲线
    plt.plot(E0_values, mean_noisy_values, marker='o', color='red', label='Mean V with Noise')  # 噪声均值曲线

    plt.title('Mean V vs E0 with Noise')
    plt.xlabel('E0')
    plt.ylabel('Mean V')
    plt.grid()
    plt.legend()  # 显示图例
    plt.show()