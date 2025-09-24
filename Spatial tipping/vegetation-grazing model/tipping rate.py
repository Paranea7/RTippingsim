import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.01
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = [1.10, 1.12, 1.13, 1.15, 1.17, 1.19]  # 不同的 c 值
num_iterations = 6000  # 迭代次数

# 定义 ω 的取值范围
omega_values = np.linspace(0.01, 2.0, 210)  # 示例：频率范围，您可以根据需要进行调整
proportions = []  # 存储每个 ω 值对应的比例


# 不随时间变化的 r
def r_static():
    return 0.47  # 静态 r 值


# 随时间变化的 r
def r(t, omega):
    return 0.47 + 0.05 * np.sin(omega * t)  # 频率为 omega 的线性变化


# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)


# RK4迭代算法的向量化实现
def rk4_step(V, r_value, V_c, c):
    k1 = (r_value * V * (1 - V / V_c) - c * V ** 2 / (V ** 2 + 1) + D * laplacian(V))
    k2 = (r_value * (V + 0.5 * dt * k1) * (1 - (V + 0.5 * dt * k1) / V_c) -
          c * (V + 0.5 * dt * k1) ** 2 / ((V + 0.5 * dt * k1) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r_value * (V + 0.5 * dt * k2) * (1 - (V + 0.5 * dt * k2) / V_c) -
          c * (V + 0.5 * dt * k2) ** 2 / ((V + 0.5 * dt * k2) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r_value * (V + dt * k3) * (1 - (V + dt * k3) / V_c) -
          c * (V + dt * k3) ** 2 / ((V + dt * k3) ** 2 + 1) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next


# 计算每个 ω 下的比例
for omega in omega_values:
    V_static = np.loadtxt('1.5-5.5.csv', delimiter=',')

    for t in range(num_iterations):
        V_static = rk4_step(V_static, r_static(), V_c, c_values[3])  # 选择第一个 c 值

    V_dynamic = np.loadtxt('1.5-5.5.csv', delimiter=',')

    for t in range(num_iterations):
        V_dynamic = rk4_step(V_dynamic, r(t, omega), V_c, c_values[3])  # 选择第一个 c 值

    V_diff = V_dynamic - V_static
    # 统计大于0的点的数量
    count_greater_than_zero = np.sum(V_diff > 0.01)
    total_points = nx * ny  # 点的总数
    proportion = count_greater_than_zero / total_points  # 比例

    proportions.append(proportion)

# 绘制比例 - ω 图
plt.figure(figsize=(10, 6))
plt.plot(omega_values, proportions, marker='o')
plt.title('Proportion of Points > 0 vs. Omega')
plt.xlabel('Omega (Frequency)')
plt.ylabel('Proportion of Points > 0')
plt.grid()
plt.show()