import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from functools import partial

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.01
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = [1.10, 1.13, 1.14, 1.15, 1.16, 1.19]  # 不同的 c 值
num_iterations = 6000  # 迭代次数
start_frame = 1600  # 提取开始帧
end_frame = 3100    # 提取结束帧

def load_static_data():
    return np.loadtxt('1.5-5.5.csv', delimiter=',')  # 预加静态数据

# 不随时间变化的 r
def r_static():
    return 0.47  # 静态 r 值

# 随时间变化的 r
def r(t):
    return 0.47 + 0.05 * np.sin(0.01 * t)

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

# 迭代和保存中间结果的函数
def run_simulation(c, V_static):
    V_dynamic = np.copy(V_static)  # 创建动态 V
    frames_static = []
    frames_dynamic = []

    for t in range(num_iterations):
        V_static = rk4_step(V_static, r_static(), V_c, c)
        V_dynamic = rk4_step(V_dynamic, r(t), V_c, c)  # 按时间更新 V_dynamic

        if start_frame <= t <= end_frame:  # 仅保留2000到4000帧
            frames_static.append(V_static)
            frames_dynamic.append(V_dynamic)

    return frames_static, frames_dynamic  # 返回静态和动态帧

# 设置动画参数
def animate(frame):
    for idx, c in enumerate(c_values):
        axs[0, idx].cla()
        axs[1, idx].cla()
        axs[2, idx].cla()

        V_static = results[idx][0][frame]  # 静态数据
        V_dynamic = results[idx][1][frame]  # 动态数据

        axs[0, idx].imshow(V_static, extent=[0, nx * dx, 0, ny * dy], origin='lower', aspect='auto', vmin=0, vmax=6)
        axs[0, idx].set_title(f'No Time: c = {c}')
        axs[0, idx].set_xlabel('x')
        axs[0, idx].set_ylabel('y')

        axs[1, idx].imshow(V_dynamic, extent=[0, nx * dx, 0, ny * dy], origin='lower', aspect='auto', vmin=0, vmax=6)
        axs[1, idx].set_title(f'Time Var: c = {c}')
        axs[1, idx].set_xlabel('x')
        axs[1, idx].set_ylabel('y')

        V_diff = V_dynamic - V_static
        axs[2, idx].imshow(V_diff, extent=[0, nx * dx, 0, ny * dy], origin='lower', aspect='auto', vmin=-2, vmax=2)
        axs[2, idx].set_title(f'Diff: c = {c}')
        axs[2, idx].set_xlabel('x')
        axs[2, idx].set_ylabel('y')

    return []

if __name__ == "__main__":
    # 创建图像和动画
    fig, axs = plt.subplots(3, 6, figsize=(30, 15))

    # 预加载静态数据
    static_data = load_static_data()

    # 使用多进程并行计算
    with mp.Pool(processes=16) as pool:
        results = pool.map(partial(run_simulation, V_static=static_data), c_values)

    # 创建动画
    num_frames = end_frame - start_frame + 1  # 计算动画要显示的帧数
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/240, blit=False)

    # 保存为GIF
    anim.save('reaction_diffusion_animation.gif', writer='pillow')

    # 显示图形窗口
    plt.show()