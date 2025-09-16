import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------
# 参数设置（可按需调整）
# ----------------------------
k0 = 10        # k 的基础值
alpha = 1
beta = 1
dt = 0.01      # 时间步长
time_end = 5000.0  # 仿真结束时间
e = 0.01
num_steps = int(time_end / dt)  # 时间步数

# 为防止内存溢出：警告信息（可删除）
print(f"num_steps = {num_steps}, total points per trajectory = {num_steps}")
print("注意：如果内存不足或绘图非常慢，请减小 time_end 或减少初始条件数量。")

# ----------------------------
# RK4 向量化实现
# ----------------------------
def rk4_vec(x0_values, dt, num_steps):
    """
    对多个初始值同时用 RK4 求解微分方程，返回 shape=(len(x0_values), num_steps) 的数组
    x' = r(t) * x * (1 - x / k(t)) - beta * x^2 / (alpha^2 + x^2)
    其中 k(t) = k0 + 2*sin(0.01*t)
          r(t) = 0.47 + 0.05*sin(0.5*t) - e*k(t)
    """
    n0 = len(x0_values)
    x_values = np.zeros((n0, num_steps))
    x_values[:, 0] = x0_values

    for i in range(1, num_steps):
        t = i * dt
        k = k0 + 2.0 * np.sin(0.01 * t)
        r = 0.47 + 0.05 * np.sin(0.5 * t) - e * k

        x_prev = x_values[:, i - 1]

        # 计算导数函数 f(x, t)
        def f(x):
            return r * x * (1.0 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2)

        k1 = dt * f(x_prev)
        k2 = dt * f(x_prev + 0.5 * k1)
        k3 = dt * f(x_prev + 0.5 * k2)
        k4 = dt * f(x_prev + k3)

        x_values[:, i] = x_prev + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return x_values

# ----------------------------
# 初始条件与求解
# ----------------------------
# 增加初始条件密度
x0_values = np.linspace(0, 9, 181)  # 151 个初始值，从 0 到 7

# 计算演化（注意：此步可能非常耗时与耗内存）
all_x_values = rk4_vec(x0_values=x0_values, dt=dt, num_steps=num_steps)

# 时间向量
time = np.arange(0.0, time_end, dt)  # 长度应为 num_steps

# ----------------------------
# 绘图：主图 + 右上角 inset（显示 time 0-700）
# ----------------------------
plt.figure(figsize=(12, 6))

# 主图：绘制每条轨迹，按初始值是否大于 2.3 着色
for i in range(len(x0_values)):
    color = 'red' if x0_values[i] > 2.3 else 'blue'
    plt.plot(time, all_x_values[i], color=color, alpha=0.5, linewidth=0.8)

plt.xlabel('Time')
plt.ylabel('x')
plt.title('Dynamics of x over time with dynamic r(t) and k(t)')
plt.axhline(y=2.3, color='grey', linestyle='--', label='x = 2.3')
plt.grid()

# 主图图例（只用两项说明颜色含义）
plt.legend(["Initial x > 2.3", "Initial x ≤ 2.3"], loc='upper left')

# 添加右上角 inset（插图）
ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='upper right', borderpad=1)

# 插图时间区间 0 -- 700
t_max_inset = 700.0
idx_max = int(t_max_inset / dt)
if idx_max > all_x_values.shape[1]:
    idx_max = all_x_values.shape[1]

# 为了性能，在插图中下采样（例如每隔 downsample 个点画一个）
downsample = 10  # 可调整：1 表示不下采样，越大越稀疏
t_inset = time[:idx_max:downsample]

for i in range(len(x0_values)):
    color = 'red' if x0_values[i] > 2.3 else 'blue'
    ax_inset.plot(t_inset, all_x_values[i, :idx_max:downsample], color=color, alpha=0.6, linewidth=0.7)

ax_inset.set_xlim(0, t_max_inset)
ax_inset.set_title('Inset: time 0-700')
ax_inset.grid(True)
ax_inset.axhline(y=2.3, color='grey', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()