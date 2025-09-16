import numpy as np
import matplotlib.pyplot as plt

# 定义参数
k0 = 10  # k的基础值
alpha = 1
beta = 1
w = 0.02
dt = 0.01  # 步长
num_steps = int(300 / dt)  # 总步数，确保足够的迭代

# RK4 方法（向量化）
def rk4_vec(x0, dt, num_steps, phi):
    x_values = np.zeros(num_steps)  # 存储每个时间步的 x 值
    x_values[0] = x0  # 初始化 x 值

    for i in range(1, num_steps):
        t = i * dt  # 当前时间
        r = 0.47 + 0.02 * np.sin(w * t +phi)  # 动态 r 值
        k = k0 + 2 * np.sin(w * t)  # 动态 k 值

        x_prev = x_values[i - 1]

        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) -
                   (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) -
                   (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) -
                   (beta * (x_prev + k3) ** 2) / (alpha ** 2 + (x_prev + k3) ** 2))

        x_values[i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values  # 返回所有时间步的 x 值

# 设置初始条件为2.3
x0 = 2.29  # 只取初始值为2.3

# 记录临界相位和状态
phi_values = []  # 用于存储相位
states = []  # 用于存储状态
state = 0
critical_phi = 0

# 使用for循环增加相位
for phi in np.arange(0, np.pi/2., np.pi/512.):
    # 计算初始值为2.3的演化
    x_values = rk4_vec(x0=x0, dt=dt, num_steps=num_steps, phi=phi)

    # 取迭代200的结果
    output_at_200 = x_values[29999]  # 取第300次的结果
    print(x_values[29999])
    # 判断状态
    if output_at_200 > 4:
        state = 1
        # 记录临界相位
        critical_phi = phi
    elif output_at_200 < 2:
        state = 0

    # 记录相位和状态
    phi_values.append(phi)
    states.append(state)

# 输出临界相位
print(f"临界相位为: {critical_phi}")

# 绘制phi与状态的图
plt.figure(figsize=(12, 6))
plt.plot(phi_values, states, marker='o')
plt.xlabel('Phi')
plt.ylabel('State')
plt.title('State vs. Phi')
plt.axhline(y=0.5, color='grey', linestyle='--', label='State Boundary')
plt.xticks(np.arange(0, max(phi_values) + 0.5, 0.5))  # 设置x轴刻度
plt.yticks([0, 1], ['State 0', 'State 1'])  # 状态标签
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()