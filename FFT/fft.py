import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 采样频率
x = np.linspace(0, 1, 1400)
# 设置要采集的信号，y就是输入的离散数据，这里的离散数据是使用函数生成的
y = 7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x) + 3*np.cos(2*np.pi*600*x)

# 进行快速傅里叶变换,快速傅里叶变换的结果是一组复数
# 结果数据长度和原始采样信号是一样的
# fft的 振幅谱 和 相位谱,是 通过对傅里叶变换得到的复数结果，进行进一步计算得到的
# 复数的模(绝对值) 就是对应的 "振幅谱",复数所对应的角度 就是所对应的 "相位谱"
fft_y = fft(y)
print(len(fft_y))
print(fft_y)

# 采样频率(每分钟采样数)
N = 1400
x = np.arange(N)
half_x = x[range(int(N/2))]     # 取一半的区间
abs_y = np.abs(fft_y)   # 取复数的绝对值(振幅谱)
angle_y = np.angle(fft_y)   # 取复数的角度(相位谱)
normalization_y = abs_y/N   # 归一化处理
normalization_half_y = normalization_y[range(int(N/2))]    # 取一半的区间
# print(len(normalization_half_y))
# print(normalization_half_y)
plt.figure()
plt.plot(x[0:50], y[0:50])
plt.show()

plt.figure()
# plt.plot(half_x[range(int(5))], normalization_half_y[range(int(5))], 'blue')
plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边频谱(归一化)', fontsize=9, color='blue')
plt.show()