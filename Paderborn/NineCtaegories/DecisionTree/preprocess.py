import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


def preprocess(d_path, d_type, rate=[0.7, 0.2, 0.1]):
    # 读取某个txt文件内的数据
    def load_data(file_path, d_type):
        f = open(file_path)
        res = []
        # 按列读文件
        while 1:
            line = f.readline()
            if not line:
                break
            dataStr = line.split(" ")
            data = []
            # 字符数组转化为float数组
            for i in range(len(dataStr) - 1):
                data.append(float(dataStr[i]))
            # 添加数据类别 (数据打标签)
            data.append(d_type)
            # 去掉最后一个空行
            if len(data) > 100:
                res.append(data)
        return res

    # 为数据打标签
    def add_label(data):
        X = []
        Y = []
        for num in data:
            X = num[0: len(num)-1]
            Y = num[len(num)-1]
        return X, Y

    # one-hot编码
    def one_hot(train_y, test_y):
        train_y = np.array(train_y).reshape([-1, 1])
        test_y = np.array(test_y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(train_y)
        train_y = Encoder.transform(train_y).toarray()
        test_y = Encoder.transform(test_y).toarray()
        train_y = np.asarray(train_y, dtype=np.int32)
        test_y = np.asarray(test_y, dtype=np.int32)
        return train_y, test_y

    # 用训练集标准差,标准化训练集和测试集
    def scalar_stand(train_x, test_x):
        scalar = preprocessing.StandardScaler().fit(train_x)
        train_x = scalar.transform(train_x)
        test_x = scalar.transform(test_x)
        return train_x, test_x

    # 切分测试集为 测试集和验证集
    def valid_test_slice(test_x, test_y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for valid_index, test_index in ss.split(test_x, test_y):
            x_valid, x_test = test_x[valid_index], test_x[test_index]
            y_valid, y_test = test_y[valid_index], test_y[test_index]
            return x_valid, y_valid, x_test, y_test

    data = load_data(d_path, d_type)
    # 划分训练集和测试集
    train = data[0: int(len(data)*rate[0])]
    test = data[len(data)-len(train): len(data)]
    # 为数据打标签
    train_x, train_y = add_label(train)
    test_x, test_y = add_label(test)
    # 对结果进行one-hot编码
    train_y, test_y = one_hot(train_y, test_y)
    # 标准化 训练集和测试集
    train_x, test_x = scalar_stand(train_x, test_x)
    # 切分测试集 为验证集+测试集
    valid_x, valid_y, test_x, test_y = valid_test_slice(test_x, test_y)

    return data


if __name__ == "__main__":
    file_path = r'D:\WorkSpace\PyCharm\GradationProject\Paderborn\NineCtaegories\prodata'
    files = os.listdir(file_path)
    for file in files:
        file = os.path.join(file_path, file)
        data = preprocess(file, 0)
        break
    num = [0, 1, 2]
    print(num[0: 1])
