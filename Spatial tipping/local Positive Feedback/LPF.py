import numpy as np
import matplotlib.pyplot as plt

# 参数定义
D = 0.5  # 扩散系数
Lambda = 0.12  # 植被耗水率
rho = 1.0  # 最大植被增长率
BC = 10.0  # 植被承载能力
mu = 2.0  # 最大放牧率
BO = 1.0  # 半饱和常数
sigma_w = 0.01  # 土壤湿度噪声标准差
sigma_B = 0.25  # 植被生物量噪声标准差
w0 = 1.0  # 土壤湿度尺度值
B0 = 1.0  # 生物量尺度值
tau_w = 1.0  # 土壤湿度尺度时间

# 降雨量 R 的范围
R_values = [1.0, 1.5, 2.0]

# 初始化网格
size = 100
w = np.random.rand(size, size) * w0  # 初始土壤湿度
B = np.random.rand(size, size) * B0  # 初始植被生物量
dt = 0.01
for R in R_values:
    w_copy = w.copy()
    B_copy = B.copy()
    for _ in range(100):  # 模拟 100 个时间步
        # 模拟噪声
        xi_w = np.random.normal(0, sigma_w, (size, size))
        xi_B = np.random.normal(0, sigma_B, (size, size))

        # 计算土壤湿度的扩散
        w_diffusion = D * (np.roll(w_copy, 1, axis=0) + np.roll(w_copy, -1, axis=0) +
                           np.roll(w_copy, 1, axis=1) + np.roll(w_copy, -1, axis=1) -
                           4 * w_copy)

        # 计算植被生物量的扩散
        B_diffusion = D * (np.roll(B_copy, 1, axis=0) + np.roll(B_copy, -1, axis=0) +
                           np.roll(B_copy, 1, axis=1) + np.roll(B_copy, -1, axis=1) -
                           4 * B_copy)

        # 更新土壤湿度
        w_copy = w_copy + dt * (R - w_copy / tau_w - Lambda * w_copy * B_copy +
                                w_diffusion + xi_w)

        # 更新植被生物量
        B_copy = B_copy + dt * (rho * B_copy * (w_copy / w0 - B_copy / BC) /
                                (B0 + B_copy) - mu * B_copy + B_diffusion + xi_B)

    # 可视化结果
    plt.figure()
    plt.imshow(B_copy, cmap='Greens')
    plt.colorbar()
    plt.title(f"LPF Model - R = {R}")
    plt.show()