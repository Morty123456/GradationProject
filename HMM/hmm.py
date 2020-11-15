from hmmlearn import hmm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# 6个隐藏状态
n = 6
CSV_PATH = './1.csv'
data = pd.read_csv(CSV_PATH, index_col=0)
# 成交量
volume = data['volume']
# 收盘价
close = data['close']
# np.array 创建数组， np.log 计算自然对数， np.diff 计算差值 a[n]-a[n-1]
# 计算每日最高最低价格的对数差值，作为特征状态的一个指标
logDel = np.log(np.array(data['high'])) - np.log(np.array(data['low']))
# 关闭交易时的价格
logRet_1 = np.array(np.diff(np.log(close)))
# print(len(logRet_1))
# [5:] 从第五个开始取   [:-5] 除了最后五个都要取
# 计算每五日的收益的对数差，作为特征状态的一个指标
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
# print(len(logRet_5))
# 计算每五日的指数成交量的对数差，作为特征状态的一个指标
logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
# 调整特征指标的长度，保持所有的数据长度相同
# print(len(logDel), len(logRet_1), len(close))
logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]
Date = pd.to_datetime(data.index[5:])
# print(Date)
# np.column_stack 将矩阵按列合并
print(len(logDel), len(logRet_5), len(logVol_5))
# 特征状态指标：每日最高最低价格的对数差值；每五日收益的对数差；每五日成交量的对数差
A = np.column_stack([logDel, logRet_5, logVol_5])
# print(A)

# 使用高斯分布的hmm模型，full指的是使用完全协方差矩阵，里面的元素都不为零
model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=2000).fit(A)
hidden_states = model.predict(A)

# 从结果看出，红色代表在上涨，绿色表示在下跌
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (hidden_states == i)
    plt.plot_date(Date[pos], close[pos], 'o', label='hidden state %d' % i, lw=2)
    plt.legend()
# plt.show()

res = pd.DataFrame({'Date': Date, 'logReg_1': logRet_1, 'state': hidden_states}).set_index('Date')
series = res.logReg_1
templist = []
plt.figure(figsize=(25, 18))
for i in range(n):
    pos = (hidden_states == i)
    pos = np.append(1, pos[:-1]) #第二天进行买入操作
    res['state_ret%d' % i] = series.multiply(pos)
    data_i = np.exp(res['state_ret%d' % i].cumsum())
    templist.append(data_i[-1])
    plt.plot_date(Date, data_i, '-', label='hidden state %d' % i)
    plt.legend()
plt.show()

templist = np.array(templist).argsort()
long = (hidden_states == templist[-1]) + (hidden_states == templist[-2])  # 买入
short = (hidden_states == templist[0]) + (hidden_states == templist[1])  # 卖出
long = np.append(0, long[:-1])
short = np.append(0, short[:-1])

# 收益曲线图
plt.figure(figsize=(25, 18))
res['ret'] = series.multiply(long) - series.multiply(short)
plt.plot_date(Date, np.exp(res['ret'].cumsum()), 'r-')
plt.show()