import numpy as np
import matplotlib.pyplot as plt

# 鐎规矮绠熼崣鍌涙殶
k0 = 10
alpha = 1
beta = 1
dt = 0.01  # 濮濄儵鏆?
time_end = 400  # 娴犺法婀＄紒鎾存将閺冨爼妫?
num_steps = int(time_end / dt)

# 閸氭垿鍣洪崠鏍畱 RK4 閺傝纭?
def rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk):
    num_initial_conditions = len(x0_values)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 鐎涙ê鍋嶅В蹇庨嚋閸掓繂顫愰崐鐓庢躬濮ｅ繋閲滈弮鍫曟？濮濄儳娈戦悩鑸碘偓?
    x_values[:, 0] = x0_values  # 閸掓繂顫愰崠鏍ㄧ槨娑擃亜鍨垫慨瀣偓?

    for i in range(1, num_steps):
        t = i * dt  # 瑜版挸澧犻弮鍫曟？
        r = 0.47 - ampr * np.sin(w * t)  # 閸斻劍鈧?r 閸?
        k = k0 + ampk * np.sin(w * t)  # 閸斻劍鈧?k 閸?

        x_prev = x_values[:, i - 1]
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) -
                   (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) -
                   (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) -
                   (beta * (x_prev + k3) ** 2) / (alpha ** 2 + (x_prev + k3) ** 2))

        x_values[:, i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 閺囧瓨鏌婂В蹇庨嚋閸掓繂顫愰崐鑲╂畱閻樿埖鈧?

    return x_values  # 鏉╂柨娲栭幍鈧張澶婂灥婵鈧厧婀幍鈧張澶嬫闂傚瓨顒為惃鍕Ц閹?

# 婢х偛濮為崚婵嗩潗閺夆€叉閻ㄥ嫬鐦戞惔?
x0_values = np.linspace(0, 7, 151)  # 閻㈢喐鍨氶弴鏉戭樋閻ㄥ嫬鍨垫慨瀣偓?

# 閸欏倹鏆熺紒鍕値
w_values = [0.01, 0.02, 0.05, 0.1]   # w閻ㄥ嫬鈧?
ampr_values = [0.02, 0.03, 0.05]     # r閻ㄥ嫭灏熼獮?
ampk_values = [0.5, 1.0, 2.0]        # k閻ㄥ嫭灏熼獮?

# 缂佹ê鍩楅幍鈧張澶岀矋閸氬牆娴橀崓?
for w in w_values:
    for ampr in ampr_values:
        for ampk in ampk_values:
            # 鐠侊紕鐣婚幍鈧張澶婂灥婵鈧偐娈戝鏂垮
            all_x_values = rk4_vectorized(x0_values, dt, num_steps, w, ampr, ampk)

            # 鐠侊紕鐣婚棄宥囧仯閸掑棗鐭忕痪?(缁€杞扮伐閸?
            time = np.arange(0, time_end, dt)
            saddle_separatrix = (2 * np.sin(w * time) + 2.32)  # 閺嶈宓侀棁鈧憰浣藉殰鐎规矮绠熸潻娆庨嚋閸掑棗鐭忕痪?

            # 缂佹ê鍩楃紒鎾寸亯
            plt.figure(figsize=(12, 6))

            for i in range(len(x0_values)):
                # 閺嶈宓侀崚婵嗩潗閸婅偐娈戞径褍鐨柅澶嬪妫版粏澹?
                color = 'red' if x0_values[i] > 2.3 else 'blue'
                plt.plot(np.arange(0, time_end, dt), all_x_values[i], color=color, alpha=0.5)

            # 濞ｈ濮為棄宥囧仯閸掑棗鐭忕痪?
            plt.plot(time, saddle_separatrix, color='gray', linestyle='-', label='Saddle Separatrix', linewidth=2)

            # 濞ｈ濮為弽鍥╊劮閸滃本鐖ｆ０?
            plt.xlabel('Time')
            plt.ylabel('x')
            plt.title(f'Dynamics of x over time with dynamic r(t) and K(t) \n w={w}, ampr={ampr}, ampk={ampk}')
            plt.axhline(y=2.3, color='grey', linestyle='--', label='x = 2.3')  # 濞ｈ濮為崣鍌濃偓鍐殠
            plt.grid()
            plt.legend()  # 閺勫墽銇氶崶鍙ョ伐
            plt.tight_layout()  # 鐠嬪啯鏆ｉ崶鎯ц埌娴犮儵鈧倸绨查弽鍥╊劮

            # 娣囨繂鐡ㄩ崶鎯у剼
            plt.savefig(f'dynamics_w-{w}_ampr-{ampr}_ampk-{ampk}.png')
            plt.close()  # 閸忔娊妫磋ぐ鎾冲閸ユ儳鑸版禒銉╁櫞閺€鎯у敶鐎?