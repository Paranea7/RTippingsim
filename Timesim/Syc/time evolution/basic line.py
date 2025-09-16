import numpy as np
import matplotlib.pyplot as plt

# 定义参数
k0 = 10
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 400  # 仿真结束时间
num_steps = int(time_end / dt)

# 向量化的 RK4 方法
def rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk):
    num_initial_conditions = len(x0_values)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 存储每个初始值在每个时间步的状态
    x_values[:, 0] = x0_values  # 初始化每个初始值

    for i in range(1, num_steps):
        t = i * dt  # 当前时间
        r = 0.47 + ampr * np.sin(w * t)  # 动态 r 值
        k = k0 + ampk * np.sin(w * t)  # 动态 k 值

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
x0_values = np.linspace(0, 7, 151)  # 生成更多的初始值

# 参数组合
w_values = [0.01, 0.02, 0.05, 0.1]   # w的值
ampr_values = [0.02, 0.03, 0.05]     # r的振幅
ampk_values = [0.5, 1.0, 2.0]        # k的振幅

# 绘制所有组合图像
for w in w_values:
    for ampr in ampr_values:
        for ampk in ampk_values:
            # 计算所有初始值的演化
            all_x_values = rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk)

            # 计算鞍点分岐线 (示例值)
            time = np.arange(0, time_end, dt)
            saddle_separatrix = (2 * np.sin(w * time) + 2.32)  # 根据需要自定义这个分岐线

            # 绘制结果
            plt.figure(figsize=(12, 6))

            for i in range(len(x0_values)):
                # 根据初始值的大小选择颜色
                color = 'red' if x0_values[i] > 2.3 else 'blue'
                plt.plot(np.arange(0, time_end, dt), all_x_values[i], color=color, alpha=0.5)

            # 添加鞍点分岐线
            plt.plot(time, saddle_separatrix, color='gray', linestyle='-', label='Saddle Separatrix', linewidth=2)

            # 添加标签和标题
            plt.xlabel('Time')
            plt.ylabel('x')
            plt.title(f'Dynamics of x over time with dynamic r(t) and K(t) \n w={w}, ampr={ampr}, ampk={ampk}')
            plt.axhline(y=2.3, color='grey', linestyle='--', label='x = 2.3')  # 添加参考线
            plt.grid()
            plt.legend()  # 显示图例
            plt.tight_layout()  # 调整图形以适应标签

            # 保存图像
            plt.savefig(f'dynamics_w-{w}_ampr-{ampr}_ampk-{ampk}.png')
            plt.close()  # 关闭当前图形以释放内存