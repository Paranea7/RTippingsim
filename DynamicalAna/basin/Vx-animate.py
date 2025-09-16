import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# 参数定义
k = 10.0
beta = 1.0
alpha = 1.0

# 自变量 t
t = np.linspace(0, 10, 200)  # 时间从0到10,共200个点

# 势能函数 V(x)
def V(x, r):
    return (r / 2) * x ** 2 - (r / 3) * (x ** 3 / k) + beta * x - beta * alpha * np.arctan(x / alpha)

# r 随时间变化
w = 0.5  # 角频率
r = 0.47 + 0.05 * np.sin(w * t)

# 绘制动画
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(-2.5, 1)
ax.set_ylim(0, 0.2)
ax.set_xlabel('x')
ax.set_ylabel('V(x)')
ax.set_title('Potential Function V(x)')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.grid()

def animate(i):
    x_values = np.linspace(-2.5, 1, 4000)
    V_values = V(x_values, r[i])
    line.set_data(x_values, V_values)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)

plt.show()