import numpy as np
import matplotlib.pyplot as plt
# 优先使用 numba 提速，如环境不支持，可注释下面两行及相关装饰器
try:
    from numba import jit
    numba_available = True
except Exception:
    numba_available = False
    def jit(func=None, **kwargs):
        if func:
            return func
        else:
            def wrapper(f):
                return f
            return wrapper

# ---------------- 模型参数 ----------------
k0 = 10.0
alpha = 1.0
beta = 1.0
dt = 0.01
time_end = 5000.0
num_steps = int(time_end / dt)

# 参数与初始条件
w_rvalue = 0.1
w_kvalue = 0.01
r0 = 0.47
ampr = 0.05
ampk = 2.0

# 瞬态检测阈值（主调：进入目标区间的判定）
tol = 0.6 + 1e-2
safe = 2  # 与 tol 相关的“稳定通过”步数阈值；此处保持原设

# 目标区间及鲁棒性参数
target = 0.6
eps4 = 0.01          # 目标区间为 [0.4 - eps4, 0.4 + eps4]
stay_steps4 = 0        # 若大于 0，将在进入区间后继续保持 stay_steps4 步才确认落点；若设为 0，进入即记为落点

# 滑动窗口稳定判定参数
window_size = 20        # 窗口长度，单位步
sigma_max = 0.005         # 窗口内样本的标准差上限，越小越苛刻
delta_tol = 0.001          # 相邻两步之间的变化阈值，用于收敛判定
converge_steps = 30         # 在进入区间后，连续 converge_steps 步满足 |x_t - x_{t-Δt}| <= delta_tol 即可确认落点

# 初始条件：单条轨迹 [2.32]
x0_values = np.array([2.32], dtype=float)

# ---------------- 目标函数与 RK4 ----------------
@jit(nopython=True)
def rhs(x, t):
    r = r0 + ampr * np.sin(w_rvalue * t)
    k = k0 + ampk * np.sin(w_kvalue * t)
    return r * x * (1.0 - x / k) - (beta * x**2) / (alpha**2 + x**2)

@jit(nopython=True)
def rk4_step(x, t, dt):
    k1 = dt * rhs(x, t)
    k2 = dt * rhs(x + 0.5*k1, t + 0.5*dt)
    k3 = dt * rhs(x + 0.5*k2, t + 0.5*dt)
    k4 = dt * rhs(x + k3, t + dt)
    return x + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

# 主计算：针对单条轨迹，返回 traj 与 t_close04
@jit(nopython=True)
def compute_single_trajectory(x0, dt, n_steps, eps4, window_size, sigma_max,
                              delta_tol, converge_steps, stay_steps4, target, t_end):
    x = x0
    traj = np.empty(n_steps, dtype=np.float64)
    traj[0] = x

    # 窗口缓存：实现简单的滑动窗口
    window_vals = np.zeros(window_size, dtype=np.float64)
    wcnt = 0
    window_filled = False

    t_close04 = -1.0
    in_target = False
    converge_counter = 0

    # 计时变量（用于在窗口稳定后是否达到允许的最少持续步数）
    # 这里采用在进入区间后直接判断，若 stay_steps4 > 0，才需要达到持续步
    t_idx = 0
    for t_idx in range(1, n_steps):
        t = t_idx * dt
        x = rk4_step(x, t, dt)
        traj[t_idx] = x

        # 进入目标区间的初步判定
        if (not in_target) and (target - eps4 <= x <= target + eps4):
            in_target = True
            # 初始化窗口
            window_vals[:] = 0.0
            wcnt = 0
            window_filled = False
            # 将当前点也放入窗口，后续将逐步填充
            window_vals[wcnt % window_size] = x
            wcnt += 1
            if wcnt >= window_size:
                window_filled = True
            # 清空收敛计数
            converge_counter = 0
        elif in_target:
            # 向窗口中加入当前值
            window_vals[wcnt % window_size] = x
            wcnt += 1
            if wcnt >= window_size:
                window_filled = True

            # 计算窗口统计
            if window_filled:
                mean = 0.0
                var = 0.0
                # 统计量计算（循环在 numba 下可能慢，但确保可用）
                s = 0.0
                for j in range(window_size):
                    s += window_vals[j]
                mean = s / window_size
                # 方差
                ss = 0.0
                for j in range(window_size):
                    diff = window_vals[j] - mean
                    ss += diff * diff
                var = ss / window_size
                # 稳定性判定：均值落在目标区间且方差小
                if (target - eps4 <= mean <= target + eps4) and (var <= sigma_max):
                    # 额外的收敛性判断：后续若干步内收敛
                    # 注意：这里在逐步收敛，需要后续观测，因没有前一时刻的窗口对比，采用简单策略
                    converge_counter += 1
                    if converge_steps > 0:
                        if converge_counter >= converge_steps:
                            # 记为落点时间，若 stay_steps4 > 0，则需要达到该持续时间
                            t_close04 = t_idx * dt
                            break
                    else:
                        # 未设定需要额外持续步，直接记为落点
                        t_close04 = t_idx * dt
                        break
                else:
                    converge_counter = 0

            # 若不需要额外持续步，或窗口未稳定则继续下一个时间步
        # 若尚未进入目标区间，继续仿真

    # 若循环结束仍未找到稳定落点
    if t_close04 < 0.0:
        t_close04 = -1.0

    return traj, t_close04

# ---------------- 运行 ----------------
def main():
    print('>>> 计算单条轨迹：初始值 x0=2.32，寻找落在 [0.4±eps4] 的稳定时间点 t_close04')
    n_steps = num_steps
    traj_list = []
    t_close04_list = []

    for idx, x0 in enumerate(x0_values):
        traj, t_close04 = compute_single_trajectory(
            x0=x0,
            dt=dt,
            n_steps=n_steps,
            eps4=eps4,
            window_size=window_size,
            sigma_max=sigma_max,
            delta_tol=delta_tol,
            converge_steps=converge_steps,
            stay_steps4=stay_steps4,
            target=target,
            t_end=time_end
        )
        traj_list.append(traj)
        t_close04_list.append(t_close04)
        print(f'  Trajectory {idx}: x0={x0:.4f}, t_close04={t_close04:.6f}')

    # 可选：画出第一条轨迹
    if len(traj_list) > 0:
        traj0 = traj_list[0]
        t_axis = np.arange(n_steps) * dt
        plt.figure(figsize=(8,4))
        plt.plot(t_axis, traj0, lw=1.2, alpha=0.9)
        plt.xlabel('time')
        plt.ylabel('x(t)')
        plt.title('Trajectory for x0=2.32 with robust t_close04 detection')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # 输出结果汇总
    print('Summary:')
    for i, tval in enumerate(t_close04_list):
        print(f'  x0[{i}] = {x0_values[i]:.6f}, t_close04 = {tval}')

if __name__ == "__main__":
    main()