import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义函数 fx = x / (1 + x^2)
def f1(x):
    return x / (1 + x ** 2)

# 定义函数 fx = r * (1 - x / k)
def f2(x, r, k):
    return r * (1 - x / k)

# 设置 x 的范围，仅包含正数
x = np.linspace(0, 10, 400)  # 仅从 0 到 10

# 设置不同的 w 值
w_values = [0.01, 0.02, 0.2]  # 可以根据需要添加更多的 w 值
colors = ['orange', 'green', 'red', 'purple', 'blue']  # 为每个 w 值指定一种颜色

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制第一个函数
y1 = f1(x)  # 计算第一个函数的值
line1, = ax.plot(x, y1, label=r"$f(x) = \frac{x}{1+x^2}$", color='blue')

# 初始化动态绘图的线
lines = []
text_labels = []
for w, color in zip(w_values, colors):
    line, = ax.plot([], [], color=color, alpha=0.5)
    lines.append(line)
    text = ax.text(0, 0, '', fontsize=8)  # 初始化文本对象
    text_labels.append(text)

# 设置标题和标签
ax.set_title('Impact of Changing w on the Functions (x > 0)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.axhline(0, color='black', linewidth=0.5, ls='--')
ax.axvline(0, color='black', linewidth=0.5, ls='--')
ax.set_ylim(0, 1)  # 设置y轴范围
ax.set_xlim(0, 10)  # 设置x轴范围

# 添加颜色区分的图例
line_legends = [ax.plot([], [], color=color, label=f"$w={w}$")[0] for w, color in zip(w_values, colors)]
ax.legend(line_legends, w_values, title='Values of w', loc='upper right')

# 激活主要图例
ax.grid()

# 动画更新函数
def update(frame):
    for j, w in enumerate(w_values):
        r = 0.47 + 0.05 * np.sin(w * frame)  # 计算当前 r 的值
        k = 10 + 2 * np.sin(w * frame)  # 计算当前 k 的值
        y2 = f2(x, r, k)  # 计算对应的 y2 值
        lines[j].set_data(x, y2)

        # 更新文本标签
        text_labels[j].set_position((8, y2[-1]))  # 设置文本位置（x坐标固定，y坐标根据最后一个点）
        text_labels[j].set_text(f"$w={w}, r={r:.2f}, k={k:.2f}$")

    return lines + text_labels

# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, 1000, 1), blit=True)

# 显示图形
plt.show()