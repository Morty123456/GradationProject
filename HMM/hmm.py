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

# 成交金额
close = data['close']

# np.array 创建数组， np.log 计算自然对数， np.diff 计算差值 a[n]-a[n-1]
logDel = np.log(np.array(data['high'])) - np.log(np.array(data['low']))
logRet_1 = np.array(np.diff(np.log(close)))
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))

# 保持所有的数据长度相同
logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]

Date = pd.to_datetime(data.index[5:])
A = np.column_stack([logDel, logRet_5, logVol_5])

model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=2000).fit(A)
hidden_states = model.predict(A)

plt.figure(figsize=(25, 18))
for i in range(n):
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
    pos = np.append(1, pos[:-1])
    res['state_ret%d' % i] = series.multiply(pos)
    data_i = np.exp(res['state_ret%d' % i].cumsum())
    templist.append(data_i[-1])
    plt.plot_date(Date, data_i, '-', label='hidden state %d' % i)
    plt.legend()
# plt.show()

templist = np.array(templist).argsort()
long = (hidden_states == templist[-1]) + (hidden_states == templist[-2])  # 买入
short = (hidden_states == templist[0]) + (hidden_states == templist[1])  # 卖出
long = np.append(0, long[:-1])
short = np.append(0, short[:-1])

plt.figure(figsize=(25, 18))
res['ret'] = series.multiply(long) - series.multiply(short)
plt.plot_date(Date, np.exp(res['ret'].cumsum()), 'r-')
# plt.show()