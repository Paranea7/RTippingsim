import numpy as np
import matplotlib.pyplot as plt

# 瀹氫箟鍙傛暟
k0 = 10
alpha = 1
beta = 1
dt = 0.01  # 姝ラ暱
time_end = 400  # 浠跨湡缁撴潫鏃堕棿
num_steps = int(time_end / dt)

# 鍚戦噺鍖栫殑 RK4 鏂规硶
def rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk):
    num_initial_conditions = len(x0_values)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 瀛樺偍姣忎釜鍒濆鍊煎湪姣忎釜鏃堕棿姝ョ殑鐘舵€?
    x_values[:, 0] = x0_values  # 鍒濆鍖栨瘡涓垵濮嬪€?

    for i in range(1, num_steps):
        t = i * dt  # 褰撳墠鏃堕棿
        r = 0.47 + ampr * np.cos(w * t)  # 鍔ㄦ€?r 鍊?
        k = k0 + ampk * np.sin(w * t)  # 鍔ㄦ€?k 鍊?

        x_prev = x_values[:, i - 1]
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) -
                   (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) -
                   (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) -
                   (beta * (x_prev + k3) ** 2) / (alpha ** 2 + (x_prev + k3) ** 2))

        x_values[:, i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 鏇存柊姣忎釜鍒濆鍊肩殑鐘舵€?

    return x_values  # 杩斿洖鎵€鏈夊垵濮嬪€煎湪鎵€鏈夋椂闂存鐨勭姸鎬?

# 澧炲姞鍒濆鏉′欢鐨勫瘑搴?
x0_values = np.linspace(0, 7, 151)  # 鐢熸垚鏇村鐨勫垵濮嬪€?

# 鍙傛暟缁勫悎
w_values = [0.01, 0.02, 0.05, 0.1]   # w鐨勫€?
ampr_values = [0.02, 0.03, 0.05]     # r鐨勬尟骞?
ampk_values = [0.5, 1.0, 2.0]        # k鐨勬尟骞?

# 缁樺埗鎵€鏈夌粍鍚堝浘鍍?
for w in w_values:
    for ampr in ampr_values:
        for ampk in ampk_values:
            # 璁＄畻鎵€鏈夊垵濮嬪€肩殑婕斿寲
            all_x_values = rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk)

            # 璁＄畻闉嶇偣鍒嗗矏绾?(绀轰緥鍊?
            time = np.arange(0, time_end, dt)
            saddle_separatrix = (2 * np.sin(w * time) + 2.32)  # 鏍规嵁闇€瑕佽嚜瀹氫箟杩欎釜鍒嗗矏绾?

            # 缁樺埗缁撴灉
            plt.figure(figsize=(12, 6))

            for i in range(len(x0_values)):
                # 鏍规嵁鍒濆鍊肩殑澶у皬閫夋嫨棰滆壊
                color = 'red' if x0_values[i] > 2.3 else 'blue'
                plt.plot(np.arange(0, time_end, dt), all_x_values[i], color=color, alpha=0.5)

            # 娣诲姞闉嶇偣鍒嗗矏绾?
            plt.plot(time, saddle_separatrix, color='gray', linestyle='-', label='Saddle Separatrix', linewidth=2)

            # 娣诲姞鏍囩鍜屾爣棰?
            plt.xlabel('Time')
            plt.ylabel('x')
            plt.title(f'Dynamics of x over time with dynamic r(t) and K(t) \n w={w}, ampr={ampr}, ampk={ampk}')
            plt.axhline(y=2.3, color='grey', linestyle='--', label='x = 2.3')  # 娣诲姞鍙傝€冪嚎
            plt.grid()
            plt.legend()  # 鏄剧ず鍥句緥
            plt.tight_layout()  # 璋冩暣鍥惧舰浠ラ€傚簲鏍囩

            # 淇濆瓨鍥惧儚
            plt.savefig(f'dynamics_w-{w}_ampr-{ampr}_ampk-{ampk}.png')
            plt.close()  # 鍏抽棴褰撳墠鍥惧舰浠ラ噴鏀惧唴瀛?