import numpy as np
import matplotlib.pyplot as plt

# 定义动态系统的参数
k = 10
alpha = 1
beta = 1

# 定义参数变化的范围
r_values = np.linspace(0, 0.9, 10)

# 定义 net function to calculate dx/dt
def dx_dt(x, r):
    return r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2)

# 绘制 nullcline
def plot_nullclines(r):
    x = np.linspace(0, k, 300)
    y_nullcline = np.zeros_like(x)  # dy/dt = 0 的 nullcline
    x_nullcline = np.zeros_like(x)  # dx/dt = 0 的 nullcline

    # 计算 x-nullcline
    for i in range(len(x)):
        x_nullcline[i] = (r * x[i] * (1 - x[i] / k)) * (alpha ** 2 + x[i] ** 2) / beta

    plt.plot(x, x_nullcline, label=f'Nullcline for r={r}', color='blue')

# 创建绘图
plt.figure(figsize=(10, 6))

# 循环绘制不同参数下的 nullcline
for r in r_values:
    plot_nullclines(r)

plt.title('Nullclines of the system for varying r values')
plt.xlabel('Population (x)')
plt.ylabel('Nullcline Value')
plt.grid()
plt.axhline(0, color='black', lw=0.8, ls='--')
plt.axvline(0, color='black', lw=0.8, ls='--')
plt.legend()
plt.show()