import os
from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


# 文件路径，每组数据长度，数据大小，训练测试比例
def prepro(d_path, length=512, number=1000, rate=[0.5, 0.25, 0.25]):
    # 读取单个文件内的电流数据(监测了两个位置的电流，分别定义为0和1)
    def load_file(filename):
        file = loadmat(filename)
        # 把keys变为数组
        keys = list(file.keys())
        # print(keys)
        data = file[keys[3]][0][0]
        # 取到data里面的电流数据(数据被包围了很多层)
        phase_current_1 = data[2][0][1][2][0]
        phase_current_2 = data[2][0][2][2][0]
        phase_current = {}
        phase_current[0] = phase_current_1
        phase_current[1] = phase_current_2
        return phase_current

    # 读取文件夹内的文件
    def capture(path):
        filenames = os.listdir(path)
        for filename in filenames:
            # 轴承的运行条件需要是一致的，转速、负载扭矩、径向作用力
            # 先取一个文件的数据进行测试(每种测试条件下，测试了20次，取其中的第一次进行测试)
            # print(filename)
            if filename.startswith('N09_M07_F10') and filename.endswith('1_1.mat'):
                file_path = os.path.join(path, filename)
                phase_current = load_file(file_path)
                # 返回两个电流中的一个试试水
                return phase_current[0]

    # 读取data文件夹下的数据(data中的子文件夹，对应着某个状态)
    # 返回结果：K、KA、KL 三种状态下的，N09_M07_F10测试条件下的 第一组数据
    def capture_file(path):
        files = os.listdir(path)
        data = {}
        count = 0
        for file in files:
            if file.endswith('01'):
                file_path = os.path.join(path, file)
                phrase_data = capture(file_path)
                data[count] = phrase_data
                count = count + 1
        return data

    # 切割数据，把每种类型的data，分割为某段大小的数组
    def slice_enc(data, slice_rate=rate[1]+rate[2]):
        Train_samples = {}
        Test_samples = {}
        train_number = number*(1-slice_rate)
        test_number = number*slice_rate
        # 遍历各种类型的数据
        for count in range(len(data)):
            Train_sample = []
            Test_sample = []
            # 每种类型的数据，都选出 number 组训练数据，每组数据的长度是 length
            slice_data = data[count]
            data_size = len(slice_data)
            # 选择训练集
            for i in range(int(train_number)):
                random_start = np.random.randint(low=0, high=(data_size-length))
                sample = slice_data[random_start: random_start+length]
                Train_sample.append(sample)
            # 选择测试集和验证集
            for i in range(int(test_number)):
                random_start = np.random.randint(low=0, high=data_size-length)
                sample = slice_data[random_start: random_start+length]
                Test_sample.append(sample)
            Train_samples[count] = Train_sample
            Test_samples[count] = Test_sample
        return Train_samples, Test_samples

    # 打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in range(len(train_test)):
            x = train_test[i]
            X += x
            Y += [label]*len(train_test[i])
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    # 用训练集标准差 标准化训练集 和 测试集
    def scalar_stand(Train_X, Test_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    # 测试集切分为 测试集和验证集
    def valid_test_slice(test_x, test_y):
        test_size = rate[2]/(rate[1]+rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for valid_index, test_index in ss.split(test_x, test_y):
            x_valid, x_test = test_x[valid_index], test_x[test_index]
            y_valid, y_test = test_y[valid_index], test_y[test_index]
            return x_valid, y_valid, x_test, y_test

    # 读数据，data是一个二维数组
    data = capture_file(d_path)
    # 制造数据，返回的也都是三维数组
    train, test = slice_enc(data)
    print(len(train), len(train[0]))
    # 打标签，返回的是二维数组
    train_x, train_y = add_labels(train)
    test_x, test_y = add_labels(test)
    # one-hot标签
    train_y, test_y = one_hot(train_y, test_y)
    # 标准化 训练集和测试集
    train_x, test_x = scalar_stand(train_x, test_x)
    valid_x, valid_y, test_x, test_y = valid_test_slice(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == "__main__":
    filename = r'D:\Data\Paderborn\data'
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepro(filename)
    print(len(train_x), len(train_y), len(valid_x), len(valid_y), len(test_x), len(test_y))
# filename = r'D:\Data\Paderborn\K001\N09_M07_F10_K001_1.mat'
# file = loadmat(filename)
# print(file.keys())
# data = file["N09_M07_F10_K001_1"][0][0]
# print(len(data))
# print(len(data[2][0]))
# print(data[2][0])
# print(len(data[2][0][1]))
# print(data[2][0][1])
# print(len(data[2][0][1][2][0]))
# print(data[2][0][1][2][0])
# print(len(data[2][0][2][2][0]))
# print(data[2][0][2][2][0])
