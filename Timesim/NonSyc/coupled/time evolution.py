import numpy as np
import matplotlib.pyplot as plt

# 定义参数
k0 = 10  # k的基础值
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
e=0.01
num_steps = int(time_end / dt)

# RK4 方法（向量化）
def rk4_vec(x0_values, dt, num_steps):
    x_values = np.zeros((len(x0_values), num_steps))  # 存储每个时间步的 x 值
    x_values[:, 0] = x0_values  # 初始化 x 值

    for i in range(1, num_steps):
        t = i * dt  # 当前时间
        k = k0 + 2 * np.sin(0.01 * t)  # 动态 k 值
        r = 0.47 + 0.05 * np.sin(0.5 * t) - e * k  # 动态 r 值

        x_prev = x_values[:, i - 1]

        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) -
                   (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) -
                   (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) -
                   (beta * (x_prev + k3) ** 2) / (alpha ** 2 + (x_prev + k3) ** 2))

        x_values[:, i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values  # 返回所有时间步的 x 值

# 增加初始条件的密度
x0_values = np.linspace(0, 7, 151)  # 生成更多的初始值

# 计算每个初始值的演化
all_x_values = rk4_vec(x0_values=x0_values, dt=dt, num_steps=num_steps)

# 绘制结果
plt.figure(figsize=(12, 6))

for i in range(len(x0_values)):
    # 根据初始值的大小选择颜色
    color = 'red' if x0_values[i] > 2.3 else 'blue'
    plt.plot(np.arange(0, time_end, dt), all_x_values[i], color=color, alpha=0.5)

# 添加标签和标题
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Dynamics of x over time with dynamic r(t) and k(t)')
plt.axhline(y=2.3, color='grey', linestyle='--', label='x = 2.3')  # 添加参考线
plt.grid()

# 添加图例
plt.legend(["Initial x > 2.3", "Initial x ≤ 2.3"], loc='upper left')
plt.tight_layout()  # 调整图形以适应标签
plt.show()