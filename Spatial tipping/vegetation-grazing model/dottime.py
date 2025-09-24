import numpy as np
import matplotlib.pyplot as plt

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
num_iterations = 50000  # 迭代次数

# 不随时间变化的 r
def r_static():
    return 0.47  # 静态 r 值

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

# 初始化网格和选择的点
V = np.loadtxt('1.5-5.5.csv', delimiter=',')
point_x, point_y = 20, 20  # 例：选择网格中心点

# 存储该点的值随时间的变化
point_values = []

# 迭代计算
for t in range(num_iterations):
    V = rk4_step(V, r_static(), V_c,1.14)
    point_values.append(V[point_x, point_y])  # 记录选择点的值

# 绘制该点随时间变化的图
plt.figure(figsize=(10, 5))
plt.plot(np.arange(num_iterations) * dt, point_values, label='Value at (25, 25)')
plt.title('Value of Point (25, 25) Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()