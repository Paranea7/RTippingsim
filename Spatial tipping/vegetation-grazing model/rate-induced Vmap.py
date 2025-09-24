import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = [1.10, 1.12, 1.13, 1.15, 1.17, 1.19]  # 不同的c值
num_iterations = 15000  # 迭代次数


def r(t):
    return 0.47 + 0.05 * np.sin(0.2*t)  # 随时间 t 线性变化

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化实现
def rk4_step(V, V_c, c, t):
    r_value = r(t)  # 获取当前时刻的 r 值
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

# 创建图像
fig, axs = plt.subplots(1, 6, figsize=(30, 5))

# 计算每个(c)组合下的结果并绘图
for idx, c in enumerate(c_values):
    V = np.random.uniform(2.3, 4.0, (nx, ny))  # 随机初始化V
    for t in range(num_iterations):  # 迭代若干次
        V = rk4_step(V, V_c, c, t)

    im = axs[idx].imshow(V, extent=[0, nx * dx, 0, ny * dy], origin='lower',
                         aspect='auto', vmin=0, vmax=5)
    axs[idx].set_title(f'c = {c}')
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('y')

# 添加颜色条，使用 Normalize 来设置范围
norm = Normalize(vmin=0, vmax=5)
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, norm=norm)

plt.show()