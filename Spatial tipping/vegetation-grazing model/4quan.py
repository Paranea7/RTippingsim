import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 2  # 网格维度
c_values = [1.10, 1.13, 1.14, 1.15, 1.16, 1.19]  # 不同的 c 值
num_iterations = 50000  # 迭代次数

# 不随时间变化的 r
def r_static():
    return 0.47  # 静态 r 值

# 随时间变化的 r
def r(t):
    return 0.47 + 0.05 * np.sin(0.01 * t) # 随时间 t 线性变化

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


# 创建图像，三行布局
fig, axs = plt.subplots(3, 6, figsize=(30, 15))

# 第一行：无时间变化的 r
for idx, c in enumerate(c_values):
    # 从 CSV 文件读取 V
    V = np.loadtxt('1.5-5.5quan.csv', delimiter=',')

    for t in range(num_iterations):  # 迭代若干次
        V = rk4_step(V, r_static(), V_c, c)

    im = axs[0, idx].imshow(V, extent=[0, nx * dx, 0, ny * dy], origin='lower',
                            aspect='auto', vmin=0, vmax=6)
    axs[0, idx].set_title(f'No Time: c = {c}')
    axs[0, idx].set_xlabel('x')
    axs[0, idx].set_ylabel('y')

# 添加公共颜色条：第一行
norm1 = Normalize(vmin=0, vmax=6)
cbar1 = fig.colorbar(im, ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04, norm=norm1)

# 第二行：随时间变化的 r
for idx, c in enumerate(c_values):
    # 从 CSV 文件读取 V
    V = np.loadtxt('1.5-5.5quan.csv', delimiter=',')

    for t in range(num_iterations):  # 迭代若干次
        V = rk4_step(V, r(t), V_c, c)

    im = axs[1, idx].imshow(V, extent=[0, nx * dx, 0, ny * dy], origin='lower',
                            aspect='auto', vmin=0, vmax=6)
    axs[1, idx].set_title(f'Time Var: c = {c}')
    axs[1, idx].set_xlabel('x')
    axs[1, idx].set_ylabel('y')

# 添加公共颜色条：第二行
norm2 = Normalize(vmin=0, vmax=6)
cbar2 = fig.colorbar(im, ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04, norm=norm2)

# 第三行：计算差值并绘制
for idx, c in enumerate(c_values):
    # 从 CSV 文件读取 V
    V_static = np.loadtxt('1.5-5.5quan.csv', delimiter=',')

    for t in range(num_iterations):
        V_static = rk4_step(V_static, r_static(), V_c, c)

    # 从 CSV 文件读取 V
    V_dynamic = np.loadtxt('1.5-5.5quan.csv', delimiter=',')

    for t in range(num_iterations):
        V_dynamic = rk4_step(V_dynamic, r(t), V_c, c)

    V_diff = V_dynamic - V_static

    im = axs[2, idx].imshow(V_diff, extent=[0, nx * dx, 0, ny * dy], origin='lower',
                            aspect='auto', vmin=-2, vmax=2)
    axs[2, idx].set_title(f'Diff: c = {c}')
    axs[2, idx].set_xlabel('x')
    axs[2, idx].set_ylabel('y')

# 添加公共颜色条：第三行
norm3 = Normalize(vmin=-2, vmax=2)
cbar3 = fig.colorbar(im, ax=axs[2, :], orientation='vertical', fraction=0.02, pad=0.04, norm=norm3)

plt.show()