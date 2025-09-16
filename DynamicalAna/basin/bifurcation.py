import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

# 定义参数
k = 10
alpha = 1
beta = 1
time_end = 100  # 仿真结束时间
dt = 0.001  # 时间步长

# 定义 ODE 系统
def odefunc(t, x, r):
    return r * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2)

# 使用 scipy 求解 ODE
def compute_steady_state(x0, r_values):
    steady_states = []
    for r in r_values:
        # 更细化的时间点
        sol = solve_ivp(odefunc, [0, time_end], [x0], args=(r,), t_eval=np.arange(0, time_end, dt))
        # 取最后一个时刻的值作为稳态
        steady_states.append(sol.y[0, -1])
    return np.array(steady_states)

# 生成不同变化速度的 r 值（包括正弦变化）
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

# 修改变化类型并计算稳态 x
change_types = ['linear', 'fast-sin', 'sin', 'slow-linear']
r_values_list = [generate_r_values(0, 1, int(1 / 0.001) + 1, ct) for ct in change_types]

# 使用并行计算求解 ODE 的结果
results = Parallel(n_jobs=-1)(delayed(compute_steady_state)(0.1, r_values) for r_values in r_values_list)

# 将平行计算的结果转为列表
x_stable_rk4_list = list(results)

# 绘制结果 - Dynamics 图
plt.figure(figsize=(12, 6))
markers = ['o', 's', 'x', '+']  # 不同的标记
for i, steady_x in enumerate(x_stable_rk4_list):
    plt.plot(r_values_list[i], steady_x, marker=markers[i], label=f'Steady state x ({change_types[i]})', linestyle='-', alpha=0.7)

plt.xlabel('r')
plt.ylabel('Steady state x')
plt.title('Dynamics of Steady State x vs r with Different Change Rates')
plt.xticks(np.arange(0, 1.1, 0.001))  # 设置 x 轴刻度
plt.yticks(np.arange(0, 11, 1))  # 根据需要调整 y 轴刻度
plt.xlim(0.56, 0.57)
plt.legend()
plt.grid()
plt.show()