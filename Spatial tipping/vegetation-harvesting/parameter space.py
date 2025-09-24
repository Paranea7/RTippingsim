import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度

# 定义 r 和 c 的取值范围
r_values = np.linspace(5, 15, 11)  # r 的取值范围
c_values = np.linspace(28, 32, 41)  # c 的取值范围

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化
def rk4_step(V, r, c):
    # 更新 k 的计算以匹配新的动态方程
    k1 = (r * V * (1 - V / V_c) - c * V / (V + 1) + D * laplacian(V))
    k2 = (r * (V + 0.5 * dt * k1) * (1 - (V + 0.5 * dt * k1) / V_c) -
          c * (V + 0.5 * dt * k1) / ((V + 0.5 * dt * k1) + 1) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r * (V + 0.5 * dt * k2) * (1 - (V + 0.5 * dt * k2) / V_c) -
          c * (V + 0.5 * dt * k2) / ((V + 0.5 * dt * k2) + 1) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r * (V + dt * k3) * (1 - (V + dt * k3) / V_c) -
          c * (V + dt * k3) / ((V + dt * k3) + 1) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next

# 计算每个(r, c)组合下的结果
def compute_for_r_c(params):
    r, c = params
    V = np.random.uniform(10, 10.1, (nx, ny))  # 每次重新初始化V，生成在10到10.1之间的均匀随机数
    for t in range(15000):  # 迭代15000次，时间步数
        V = rk4_step(V, r, c)
    return np.mean(V)  # 返回最终的平均V

# 创建参数组合
params = [(r, c) for r in r_values for c in c_values]

# 使用多进程计算不同(r, c)组合得到的V
if __name__ == '__main__':
    with Pool() as pool:  # 使用全部可用的CPU核心
        mean_values = pool.map(compute_for_r_c, params)

    # reshape mean_values 为 2D数组
    mean_values = np.array(mean_values).reshape(len(r_values), len(c_values))

    # 绘制 r-c 图
    plt.figure(figsize=(10, 6))
    plt.contourf(c_values, r_values, mean_values, levels=50, cmap='viridis')
    plt.colorbar(label='Mean V')
    plt.title('Parameter Space r-c Diagram')
    plt.xlabel('c')
    plt.ylabel('r')
    plt.grid()

    # 保存图像
    plt.savefig('r_c_diagram.png', dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
    plt.show()  # 显示图像