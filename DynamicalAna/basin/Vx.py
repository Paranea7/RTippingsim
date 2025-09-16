import numpy as np
import matplotlib.pyplot as plt

# 参数定义
r = 0.42
k = 10.0
beta = 1.0
alpha = 1.0

# 势能函数 V(x)
def V(x, r):
    return (r / 2) * x ** 2 - (r / 3) * (x ** 3 / k) + beta * x - beta * alpha * np.arctan(x / alpha)

# 生成 x 轴数值
x_values = np.linspace(-2.5, 1, 4000)
V_values = V(x_values, r)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_values, V_values, label='Potential Function V(x)')
plt.title("Potential Function")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()