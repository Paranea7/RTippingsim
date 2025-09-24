import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 参数设置
r = 10.0
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = np.linspace(28, 32, 41)  # 从25到27，生成21个c值

# 初始化网格和随机初始V
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化
def rk4_step(V, c):
    k1 = (r * V * (1 - V / V_c) - c * V / (V + 1) + D * laplacian(V))
    k2 = (r * (V + 0.5 * dt * k1) * (1 - (V + 0.5 * dt * k1) / V_c) -
          c * (V + 0.5 * dt * k1) / (V + 0.5 * dt * k1 + 1) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r * (V + 0.5 * dt * k2) * (1 - (V + 0.5 * dt * k2) / V_c) -
          c * (V + 0.5 * dt * k2) / (V + 0.5 * dt * k2 + 1) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r * (V + dt * k3) * (1 - (V + dt * k3) / V_c) -
          c * (V + dt * k3) / (V + dt * k3 + 1) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next

# 计算每个c值下的结果
def compute_for_c(c):
    V = np.random.uniform(10, 10.1, (nx, ny))  # 每次重新初始化V，生成在10到10.1之间的均匀随机数
    for t in range(15000):  # 迭代15000次，时间步数
        V = rk4_step(V, c)
    return V  # 返回最终的V

# 使用多进程计算不同$c$值得到的V
if __name__ == '__main__':
    with Pool() as pool:  # 使用全部可用的CPU核心
        results = pool.map(compute_for_c, c_values)

    # 计算均值
    mean_values = [np.mean(result) for result in results]  # 原始均值计算
    # 添加高斯白噪声
    noise_level = 1.0  # 噪声强度
    noisy_results = [result + np.random.normal(0, noise_level, result.shape) for result in results]
    mean_noisy_values = [np.mean(noisy_result) for noisy_result in noisy_results]  # 噪声版本均值计算

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, mean_values, marker='o', label='Original Mean V')  # 原始均值曲线
    plt.plot(c_values, mean_noisy_values, marker='o', color='red', label='Mean V with Noise')  # 噪声均值曲线

    plt.title('Mean V vs c with Noise')
    plt.xlabel('c')
    plt.ylabel('Mean V')
    plt.grid()
    plt.legend()  # 显示图例
    plt.show()