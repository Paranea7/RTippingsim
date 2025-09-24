import numpy as np
import matplotlib.pyplot as plt

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.001
dx = dy = 0.1
nx = ny = 50  # 网格维度
num_iterations = 60000  # 迭代次数
c_values = 1.14

# 记录每个r值对应的浓度
r_values = np.linspace(0, 10, 100)
concentrations = []

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

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

# 扫描 r 值
for r in r_values:
    V = np.full((nx, ny), 1.0)  # 设定初始浓度, 可改变初始值探测不同情形
    for t in range(num_iterations):
        V = rk4_step(V, r, V_c, c_values)

    concentrations.append(V[25, 25])  # 记录 (25, 25) 的浓度

# 画出 r - V_c 图
plt.figure(figsize=(10, 6))
plt.plot(r_values, concentrations, label='V at (25, 25)', color='blue')
plt.xlabel('r Values')
plt.ylabel('Concentration at (25, 25)')
plt.title('r - V_c Plot')

# 假设我们根据一些条件来判断单稳态与双稳态区域
# 这只是一个示例条件，您需要用适合您系统性质的判断公式来代替
for i in range(1, len(concentrations) - 1):
    if concentrations[i-1] < concentrations[i] and concentrations[i+1] < concentrations[i]:
        plt.axvline(x=r_values[i], color='red', linestyle='--', label='Possible Bistability Zone')

plt.legend()
plt.show()