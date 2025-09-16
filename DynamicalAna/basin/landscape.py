import numpy as np
import matplotlib.pyplot as plt

# 参数
r = 0.47  # 增长率
k = 10   # 环境承载能力
alpha = 1.0
beta = 1.0

# 定义函数
def dynamics(x):
    return r * x * (1 - x / k) - (alpha**2 * x**2) / (x**2 + beta**2)

# 创建 x 的值
x_values = np.linspace(0, 15, 400)
y_values = dynamics(x_values)

# 绘制景观图
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=r'$\frac{dx}{dt}$', color='blue')
plt.axhline(0, color='black', lw=0.5, linestyle='--')

plt.scatter([1, 2], [0, 0], color='red')

plt.title('Landscape of the Dynamical System')
plt.xlabel('x')
plt.ylabel(r'$\frac{dx}{dt}$')
plt.grid()
plt.legend()
plt.show()