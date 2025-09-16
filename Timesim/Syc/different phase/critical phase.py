import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 参数定义
# ----------------------------
k0 = 10        # k 的基础值，k(t) = k0 + 2*sin(w*t)
alpha = 1      # 方程中阻尼项的分母参数 alpha^2 + x^2
beta = 1       # 方程中非线性项的系数
w = 0.02       # 驱动频率（用于 r(t) 和 k(t) 的正弦项）
dt = 0.01      # 时间步长
num_steps = int(500 / dt)  # 总步数：模拟时间 0..500，对应的步数 = 500 / dt
# 说明：若修改 dt 或总时间，请同步调整 num_steps

# ----------------------------
# RK4 积分器：对单个初始值 x0 在给定相位 phi 下积分
# 返回长度为 num_steps 的数组 x_values，记录每个时间步的 x
# ----------------------------
def rk4_vec(x0, dt, num_steps, phi):
    """
    使用经典 RK4 方法对微分方程积分：
    x' = r(t)*x*(1 - x/k(t)) - beta*x^2/(alpha^2 + x^2)
    其中 r(t) = 0.47 + 0.02 * sin(w * t + phi)
          k(t) = k0 + 2 * sin(w * t)
    x0: 初始值（标量）
    dt: 时间步长
    num_steps: 总步数
    phi: 相位偏移（影响 r(t)）
    返回：长度为 num_steps 的 numpy 数组 x_values
    """
    # 初始化结果数组并填入初始值
    x_values = np.zeros(num_steps)
    x_values[0] = x0

    # 时间步循环，从第1步（索引1）到 num_steps-1
    for i in range(1, num_steps):
        t = i * dt  # 当前时间点（与前一步 x_values[i-1] 相关）
        # 动态参数 r(t) 与 k(t)
        r = 0.47 + 0.02 * np.sin(w * t + phi)   # r 随时间和相位变化
        k = k0 + 2 * np.sin(w * t)              # k 随时间变化（但不受 phi 影响）

        x_prev = x_values[i - 1]  # 取上一步的 x

        # 计算 RK4 的四个斜率增量（k1..k4 这里仅为局部变量名）
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        # 对 k2,k3,k4 使用相同的微分方程形式，但以中间值估计 x
        x_temp = x_prev + 0.5 * k1
        k2 = dt * (r * x_temp * (1 - x_temp / k) - (beta * x_temp ** 2) / (alpha ** 2 + x_temp ** 2))

        x_temp = x_prev + 0.5 * k2
        k3 = dt * (r * x_temp * (1 - x_temp / k) - (beta * x_temp ** 2) / (alpha ** 2 + x_temp ** 2))

        x_temp = x_prev + k3
        k4 = dt * (r * x_temp * (1 - x_temp / k) - (beta * x_temp ** 2) / (alpha ** 2 + x_temp ** 2))

        # RK4 更新公式
        x_values[i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return x_values

# ----------------------------
# 主程序：设置初始条件并扫描不同的相位 phi
# ----------------------------
# 初始值设为 2.29（注释提到 2.3，但实际代码中用了 2.29）
x0 = 2.29

# 用于保存 phi 与对应状态（state = 0 或 1）
phi_values = []
states = []

# state 的默认值（若没有满足判定条件会保持上一次的值）
state = 0
critical_phi = None  # 用于记录“临界相位”，初始化为 None（未找到）

# 说明：下面的 phi 刻度使用 np.arange(0, pi/2, pi/512)
# 实际上 (pi/2) / (pi/512) = 256，因此步骤数大约为 256（终点开区间）。
# 若你想要精确的 N 个点，建议使用 np.linspace。
phi_start = 0.0
phi_end = np.pi / 2.0
phi_step = np.pi / 1024.0

# 我们想在某个指定时间点读取 x 的值来判定状态。原代码写“迭代200”，
# 但使用的索引是 29999，这不一致。这里我把注释与索引都显式化：
# 如果你想读第 30000 步（从0开始），对应的时间约为 30000*dt = 300。
# 如果想读第 200 步，对应索引应该是 199，时间约为 1.99。
# 下面我保留原索引（29999），但增加检查确保不会索引越界。
desired_index = 49999  # 原代码使用的索引（对应 t ≈ 299.99）
if desired_index >= num_steps:
    raise ValueError(f"desired_index ({desired_index}) 超过 num_steps ({num_steps})；请增大 num_steps 或减小 desired_index。")

# 遍历 phi 并记录状态
for phi in np.arange(phi_start, phi_end, phi_step):
    # 调用 RK4 积分器得到该 phi 下的时间序列 x(t)
    x_values = rk4_vec(x0=x0, dt=dt, num_steps=num_steps, phi=phi)

    # 取指定索引处的值作为判定依据（此处为 x_values[29999]，对应 t ≈ 300）
    output_at_desired = x_values[desired_index]
    # 注意：原程序每次打印这个值会产生大量输出，这里保留一次性打印（可注释掉）
    print(f"phi={phi:.6f}, x(t_index={desired_index})={output_at_desired:.6f}")

    # 判定状态：
    # 若输出大于4 -> state = 1（高值态）
    # 若输出小于2 -> state = 0（低值态）
    # 若输出落在 [2,4] 区间 -> 保持上一次的 state（与原程序一致）
    if output_at_desired > 4:
        state = 1
        # 记录临界相位：这里按原逻辑每次满足都会更新 critical_phi，
        # 因此最终得到的是最后一个满足条件的 phi，而不是第一个。
        # 如果你希望记录第一个使状态变为1的 phi，请在记录后 break 出循环或只在 critical_phi is None 时赋值。
        critical_phi = phi
    elif output_at_desired < 2:
        state = 0
    # else: 若在 [2,4]，state 保持不变

    # 记录本次 phi 与对应的 state
    phi_values.append(phi)
    states.append(state)

# 循环结束后输出“临界相位”
print(f"临界相位（最后一个满足 x>4 的 phi）为: {critical_phi}")

# ----------------------------
# 绘图：phi vs state
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(phi_values, states, marker='o', linestyle='-')
plt.xlabel('Phi')
plt.ylabel('State')
plt.title('State vs. Phi')

# 画一条横线作为状态判定的视觉参考线（0.5）
plt.axhline(y=0.5, color='grey', linestyle='--', label='State Boundary')

# 设置 x 轴刻度：从 0 到最大 phi，步长 0.5（弧度单位）
# 如果 phi 最大值小于某阈值，这个 xticks 可能会生成稀疏刻度；可以按需修改
plt.xticks(np.arange(0, max(phi_values) + 0.5, 0.5))

# y 轴只显示两种状态标签
plt.yticks([0, 1], ['State 0', 'State 1'])

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()