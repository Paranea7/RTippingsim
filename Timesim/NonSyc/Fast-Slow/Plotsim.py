import numpy as np
import matplotlib.pyplot as plt
import numba

# 定义参数（保留原有常量）
k0 = 10.0
alpha = 1.0
beta = 1.0
dt = 0.01  # 步长
time_end = 5000  # 仿真结束时间
num_steps = int(time_end / dt)

# 取消绘制基线的设定：我们不再绘制 axhline 参考线
# 你也可以在需要时手动添加其他标注，但这里移除了绘制基线的部分

# w_rvalue 与 w_kvalue 的具体数值数组
w_rvalue_list = [0.1, 0.2, 0.5]   # r 的变化角频率数组（单位：弧度/时间单位）
w_kvalue_list = [0.01, 0.02, 0.05]  # k 的变化角频率数组（单位：弧度/时间单位）

# r 的基线与振幅、k 的振幅
r0 = 0.47
ampr = 0.02
ampk = 1.0

# 使用 Numba 加速的 RK4 方法
@numba.jit(nopython=True)
def rk4_vectorized(x0_values, dt, num_steps, r0, w_rvalue, ampr, k0, w_kvalue, ampk):
    num_initial_conditions = len(x0_values)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 存储每个初始值在每个时间步的状态
    x_values[:, 0] = x0_values  # 初始化每个初始值

    for i in range(1, num_steps):
        t = i * dt  # 当前时间
        # 动态 r 与 k 的计算（基于时间的正弦变化）
        r = r0 + ampr * np.sin(w_rvalue * t)
        k = k0 + ampk * np.sin(w_kvalue * t)

        x_prev = x_values[:, i - 1]
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) -
                   (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) -
                   (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) -
                   (beta * (x_prev + k3) ** 2) / (alpha ** 2 + (x_prev + k3) ** 2))

        x_values[:, i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 更新每个初始值的状态

    return x_values  # 返回所有初始值在所有时间步的状态

# 增加初始条件的密度
x0_values = np.linspace(0, 9, 910)  # 生成更多的初始值

# 参数组合：不再使用外层的 w_values，直接遍历 w_rvalue_list 与 w_kvalue_list 的组合
ampr_values = [0.02, 0.03, 0.05]     # r 的振幅
ampk_values = [1.0, 2.0]             # k 的振幅（保持多样性）

# 预计算时间向量用于绘制和裁剪
time = np.arange(0, time_end, dt)

# 设置 0-700 时间窗口的索引范围
t_window_end = 700
idx_end_window = min(len(time) - 1, int(t_window_end / dt))
time_window = time[:idx_end_window + 1]

# 绘制所有组合图像并保存
for r_index, w_rvalue in enumerate(w_rvalue_list):
    for k_index, w_kvalue in enumerate(w_kvalue_list):
        for ampr_index, ampr in enumerate(ampr_values):
            for ampk_index, ampk in enumerate(ampk_values):
                # 计算所有初始值的演化
                all_x_values = rk4_vectorized(x0_values, dt, num_steps, r0, w_rvalue, ampr, k0, w_kvalue, ampk)

                # 这里不绘制基线，且加入一个示意性分岔线（如果需要可自行移除或替换）
                saddle_separatrix = np.sin(w_rvalue * time) + 2.36  # 仅用于可视化参考

                # 创建新的图形
                plt.figure(figsize=(10, 6))

                # 主图：在 0 到 time_end 的全局时间轴绘制所有初始值
                for i in range(len(x0_values)):
                    color = 'red' if x0_values[i] > 2.36 else 'blue'
                    plt.plot(time, all_x_values[i], color=color, alpha=0.5)

                # 添加鞍点分岌线（示例）
                plt.plot(time, saddle_separatrix, color='gray', linestyle='-', label='Saddle Separatrix', linewidth=2)

                # 添加标签和标题
                plt.xlabel('Time')
                plt.ylabel('x')
                plt.title(f'w_rvalue={w_rvalue}, w_kvalue={w_kvalue}  --  ampr={ampr}, ampk={ampk}')
                plt.grid()
                plt.legend()  # 显示图例

                # 创建右上角的小窗截取 0-700 的时间段
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(plt.gca(), width="28%", height="28%", loc="upper right",
                                      bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure,
                                      borderpad=0.2)

                for i in range(len(x0_values)):
                    plt.plot(time_window, all_x_values[i][:idx_end_window + 1], color=('red' if x0_values[i] > 2.36 else 'blue'),
                             alpha=0.5)

                plt.plot(time_window, saddle_separatrix[:idx_end_window + 1], color='gray', linestyle='-', linewidth=1.5)

                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.set_title('0-700 window', fontsize=8)

                # 保存图像
                plt.savefig(f'dynamics_plot_r{r_index}_k{k_index}_ampr{ampr}_ampk{ampk}.png', dpi=150, bbox_inches='tight')
                plt.close()  # 关闭当前图形以释放内存