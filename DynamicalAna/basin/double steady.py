import numpy as np
import matplotlib.pyplot as plt

# 定义常数
k = 10
alpha = 1
beta = 1
dt = 0.1
num_steps = 1000


# 定义 ODE 系统
def ode_func(t, x, r):
    return r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2)


# RK4 方法
def rk4(x0, r_value, dt, num_steps):
    # 初始化数组
    x_values = np.zeros(num_steps)
    x_values[0] = x0
    for step in range(1, num_steps):
        t = step * dt
        k1 = ode_func(t, x_values[step - 1], r_value)
        k2 = ode_func(t + dt / 2, x_values[step - 1] + k1 * dt / 2, r_value)
        k3 = ode_func(t + dt / 2, x_values[step - 1] + k2 * dt / 2, r_value)
        k4 = ode_func(t + dt, x_values[step - 1] + k3 * dt, r_value)
        x_values[step] = x_values[step - 1] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return x_values[-1]  # 返回最终的稳态值


def generate_r_values(start, end, num_points, change_type='linear'):
    if change_type == 'linear':
        return np.linspace(start, end, num_points)
    elif change_type == 'fast-sin':
        return (np.sin(np.linspace(0, 2 * np.pi, num_points)) + 1) / 2 * (end - start) + start
    elif change_type == 'sin':
        return np.sin(np.linspace(0, np.pi, num_points))  # 使用正弦函数进行快速变化
    elif change_type == 'slow-linear':
        return np.linspace(start, end, num_points) ** 0.5  # 减缓线性变化
    return None


# 定义初始条件
X_initial_conditions = np.linspace(0.1, 9, 10)  # 初始条件范围

# 定义不同的r值变化类型
r_types = ['linear', 'fast-sin', 'sin', 'slow-linear']
r_value_results = []

# 对每种变化类型进行计算
for change_type in r_types:
    r_values = generate_r_values(0.1, 1, 100)  # 生成r值
    steady_values = []

    # 对每个 r 值，从不同初始条件计算稳态 x
    for r_value in r_values:
        for x0 in X_initial_conditions:
            steady_x = rk4(x0=x0, r_value=r_value, dt=dt, num_steps=num_steps)
            steady_values.append((r_value, steady_x))  # 记录 (r, steady_x)

    r_value_results.append(np.array(steady_values))

# 绘制结果
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green', 'red']
for i, results in enumerate(r_value_results):
    r_vals_stable = results[:, 0]
    steady_vals = results[:, 1]
    plt.scatter(r_vals_stable, steady_vals, c=colors[i], label=f'Change Type: {r_types[i]}', alpha=0.5)

plt.xlabel('Parameter r')
plt.ylabel('Steady State x')
plt.title('Steady State Values Analysis with Different r Changes')
plt.axhline(0, color='gray', lw=0.5, ls='--')  # x 轴参考线
plt.legend()
plt.grid()
plt.xlim((0.1, 1))
plt.ylim((-0, 10))  # 设置 y 轴限制以便更好展示根和稳定状态
plt.show()