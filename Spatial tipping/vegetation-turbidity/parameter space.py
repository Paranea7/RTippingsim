import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 参数设置
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
p = 4.0
E0_values = np.linspace(4, 6, 21)  # 从4到6，生成21个E0值
r_values = np.linspace(5, 15, 21)  # 从5到15，生成21个r值
h_E = 1.5  # 设定的常数
h_v = 0.2  # 设定的常数

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化
def rk4_step(V, r, E0):
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

# 计算每个(E0, r)组合下的结果
def compute_for_E0_r(params):
    E0, r = params
    V = np.random.uniform(10, 10.1, (nx, ny))  # 每次重新初始化V，生成在10到10.1之间的均匀随机数
    for t in range(15000):  # 迭代15000次，时间步数
        V = rk4_step(V, r, E0)
    return np.mean(V)  # 返回最终的平均V

# 创建参数组合
params = [(E0, r) for E0 in E0_values for r in r_values]

# 使用多进程计算不同(E0, r)组合得到的均值
if __name__ == '__main__':
    with Pool() as pool:  # 使用全部可用的CPU核心
        mean_values = pool.map(compute_for_E0_r, params)

    # 重新塑形mean_values 为 2D 数组
    mean_values = np.array(mean_values).reshape(len(E0_values), len(r_values))

    # 绘制参数空间图
    plt.figure(figsize=(10, 6))
    plt.contourf(E0_values, r_values, mean_values, levels=50, cmap='viridis')  # 根据E0和r绘制图
    plt.colorbar(label='Mean V')
    plt.title('Parameter Space Diagram of E0-r')
    plt.xlabel('E0')  # 横坐标为E0
    plt.ylabel('r')   # 纵坐标为r
    plt.grid()

    # 保存图像
    plt.savefig('E0_r_diagram.png', dpi=300)  # 保存为PNG格式，分辨率为300 dpi
    plt.show()  # 显示图像