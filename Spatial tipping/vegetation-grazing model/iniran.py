import numpy as np

# 随机初始化V
V = np.random.uniform(0.1, 5.1, (50, 50))

# 保存V到CSV文件
np.savetxt('0.1-5.1.csv', V, delimiter=',')