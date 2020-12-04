import pandas as pd
import numpy as np


def prepro(path, rate):
    # 把数据按列进行归一化，全部缩放到 0,1 之间
    def NormalizeMult(data):
        # 把读到的数据变为数组，大小为 43800*7
        data = np.array(data)
        # 新建一个 2*data.shape[1]=14 大小的数组
        normalize = np.arange(2 * data.shape[1], dtype='float64')
        # reshape新生成一个 7*2 的数组，此数组和原数组共用一个内存
        normalize = normalize.reshape(data.shape[1], 2)
        for i in range(0, data.shape[1]):
            # 第i列
            list = data[:, i]
            # np.percentile(xxx, n) 找到xxx中第n%位置上的数字
            # 下面的函数就是找到list每一列中的最大值和最小值
            listlow, listhigh = np.percentile(list, [0, 100])
            normalize[i, 0] = listlow
            normalize[i, 1] = listhigh
            # 对每一列的数据进行归一化
            delta = listhigh - listlow
            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = (data[j, i] - listlow) / delta
        return data, normalize

    # 构造训练集 测试集 验证集
    # look_back 是每组数据的长度
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            # 列表切片, 取 i到i+look_back行
            dataX.append(dataset[i: i + look_back, :])
            # 列表切片, 取第 i+look_back 行
            dataY.append(dataset[i + look_back, :])
        TrainX = np.array(dataX)
        TrainY = np.array(dataY)
        return TrainX, TrainY

    data = pd.read_csv(path)
    data = data.drop(['date', 'wnd_dir'], axis=1)
    data, normalize = NormalizeMult(data)
    # 把污染数据单独拿出来，作为 43800*1 大小的数组
    pollution_data = data[:, 0].reshape(len(data), 1)
    train_X, _ = create_dataset(data, 20)
    _, train_Y = create_dataset(pollution_data, 20)
    length = len(train_X)
    train_len = int(length*rate[0])
    valid_len = int(length*rate[1])
    train_x, train_y = train_X[0: train_len], train_Y[0: train_len]
    valid_x, valid_y = train_X[train_len: train_len+valid_len], train_Y[train_len: train_len+valid_len]
    test_x, test_y = train_X[train_len+valid_len:], train_Y[train_len+valid_len:]
    # print(len(train_X), len(train_x), len(valid_x), len(test_x))
    # print(len(train_Y), len(train_y), len(valid_y), len(test_y))
    return train_x, train_y, valid_x, valid_y, test_x, test_y