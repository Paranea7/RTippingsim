import numpy as np
import matplotlib.pyplot as plt

# 参数定义
m = 0.1  # 死亡率
d = 0.1  # 退化率
c = 0.1  # 竞争强度
r = 0.01  # 自发放射率
f = 0.9  # 局部促进系数
delta = 0.1  # 种子传播率
dt = 0.5  # 时间步长

# 湿度参数 b 的范围
b_values = [0.3, 0.5, 0.8]

# 初始化网格
size = 100
grid = np.zeros((size, size), dtype=int)
# 随机初始化一些植被单元格
grid[np.random.rand(size, size) < 0.1] = 1  # 1 表示植被 (+)

for b in b_values:
    grid_copy = grid.copy()
    for _ in range(100):  # 模拟 100 个时间步
        new_grid = grid_copy.copy()
        for i in range(size):
            for j in range(size):
                # 计算周围植被的比例
                neighbors = grid_copy[max(0, i-1):min(size, i+2), max(0, j-1):min(size, j+2)]
                q_plus = np.sum(neighbors == 1) / neighbors.size
                q_0 = np.sum(neighbors == 0) / neighbors.size
                q_minus = np.sum(neighbors == -1) / neighbors.size

                # 状态转换规则
                if grid_copy[i, j] == 1:  # 植被
                    if np.random.rand() < m * dt:
                        new_grid[i, j] = 0
                elif grid_copy[i, j] == 0:  # 肥沃土壤
                    if np.random.rand() < (delta * q_plus + (1 - delta) * q_plus) * (b - c * q_plus) * dt:
                        new_grid[i, j] = 1
                    elif np.random.rand() < r * dt:
                        new_grid[i, j] = 1
                    elif np.random.rand() < d * dt:
                        new_grid[i, j] = -1
                elif grid_copy[i, j] == -1:  # 退化土壤
                    if np.random.rand() < f * q_plus * dt:
                        new_grid[i, j] = 0
        grid_copy = new_grid

    # 可视化结果
    plt.figure()
    plt.imshow(grid_copy, cmap='Greens')
    plt.colorbar()
    plt.title(f"LFCA Model - b = {b}")
    plt.show()