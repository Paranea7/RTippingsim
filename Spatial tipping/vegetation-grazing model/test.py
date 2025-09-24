import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 参数设置
r = 0.47
V_c = 10.0
D = 0.001
dt = 0.01
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = np.linspace(0, 2, 21)  # 从0到2，生成21个c值

# 初始化网格
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化
def rk4_step(V, c):
    k1 = (r * V * (1 - V / V_c) - c * V ** 2 / (V ** 2 + 1) + D * laplacian(V))
    k2 = (r * (V + 0.5 * dt * k1) * (1 - (V + 0.5 * dt * k1) / V_c) -
          c * (V + 0.5 * dt * k1) ** 2 / ((V + 0.5 * dt * k1) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r * (V + 0.5 * dt * k2) * (1 - (V + 0.5 * dt * k2) / V_c) -
          c * (V + 0.5 * dt * k2) ** 2 / ((V + 0.5 * dt * k2) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r * (V + dt * k3) * (1 - (V + dt * k3) / V_c) -
          c * (V + dt * k3) ** 2 / ((V + dt * k3) ** 2 + 1) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next

# 计算稳态 x 值的函数
def compute_for_c(c, initial_V):
    V = initial_V.copy()
    for t in range(1500):  # 迭代1500次，时间步数
        V = rk4_step(V, c)
    return V  # 返回最终的 V

# 初始条件设置
initial_V_increase = np.random.uniform(6.0, 10.0, size=(ny, nx))  # 从0.1到5.1的均匀随机数
initial_V_decrease = np.random.uniform(0.1, 5.1, size=(ny, nx))  # 从0.1到5.1的均匀随机数

# 计算均值
mean_values_increase = []
mean_values_decrease = []

# 计算增加 c 值情况下的 V
for c in c_values:
    V_increase = compute_for_c(c, initial_V_increase)
    mean_values_increase.append(np.mean(V_increase))  # 计算均值并存储

# 计算减少 c 值情况下的 V
reversed_c_values = np.flip(c_values)  # 倒过来的 c 值
for c in reversed_c_values:
    V_decrease = compute_for_c(c, initial_V_decrease)
    mean_values_decrease.append(np.mean(V_decrease))  # 计算均值并存储

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(c_values, mean_values_increase, marker='o', label='Mean V (Increasing c)')  # 原始均值曲线
plt.plot(reversed_c_values, mean_values_decrease, marker='o', color='red', label='Mean V (Decreasing c)')  # 反向均值曲线

plt.title('Mean V vs c')
plt.xlabel('c')
plt.ylabel('Mean V')
plt.grid()
plt.legend()  # 显示图例
plt.show()