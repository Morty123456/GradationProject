import os
from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

# 文件路径, 每组数据长度, 数据大小, 训练验证测试比例
def preprocess(d_path, length=1800, number=1000, rate=[0.7, 0.2, 0.1]):

    # 加载此路径下的文件
    # 总共有九种类别
    # 每组类别下都只读取N09_M07_F10测试条件下的数据
    def capture(path):
        folders = os.listdir(path)
        data = {}
        count = 0
        # print(folders)
        for folder in folders:
            folder_path = os.path.join(path, folder)
            phrase_data = capture_folder(folder_path)
            data[count] = phrase_data
            count += 1
        # print(len(data))
        return data

    # 读取此文件夹中的数据
    def capture_folder(path):
        filenames = os.listdir(path)
        phase_data = {}
        count = 0
        for filename in filenames:
            # print(filename)
            # 轴承的运行条件需要是一致的: 转速、负载扭矩、径向作用力
            # 每个轴承的运行状态设置了四种, 每种条件下测试了20次
            if filename.startswith('N09_M07_F10'):
                file_path = os.path.join(path, filename)
                # print(file_path)
                data = load_file(file_path)
                # 先返回监测的两个电流数据中的一个,试试效果
                phase_data[count] = data[0]
                count += 1
        # print(len(phase_data))
        return phase_data

    def load_file(file_path):
        file_data = loadmat(file_path)
        # 把数据的keys变为数组
        keys = list(file_data.keys())
        data = file_data[keys[3]][0][0]
        # 取到data中的电流数据
        # 电流数据取了两个位置的
        phase_current_1 = data[2][0][1][2][0]
        phase_current_2 = data[2][0][1][2][0]
        phase_current = {}
        phase_current[0] = phase_current_1
        phase_current[1] = phase_current_2
        return phase_current

    # 切分数据, 把每组电流数据, 制作若干个length长度的数组(length为一个周期的数据,分类效果较好)
    # 这次使用的是测试条件下的20组数据，所以传进来的data是三维的 [9][20][20w+]
    def slice_data(data, slice_rate=rate[1]+rate[2]):
        train_samples = {}
        test_samples = {}
        # 每种类型的数据取number组,每种类型的数据 有len(data[0])个文件，所以每个文件取ratenumber个数据
        ratenumber = int(number/len(data[0]))
        train_number = int(ratenumber*(1-slice_rate))
        for count in range(len(data)):
            train_sample = []
            test_sample = []
            for thisCount in range(len(data[count])):
                # 本次要切割的数据
                slice_data = data[count][thisCount]
                data_size = len(slice_data)
                # 选择训练集的数据
                # 从某个随机位置开始,向后取length长度的数据
                for i in range(train_number):
                    random_start = np.random.randint(low=0, high=(data_size - length))
                    sample = slice_data[random_start: random_start + length]
                    train_sample.append(sample)
                for i in range(ratenumber - train_number):
                    random_start = np.random.randint(low=0, high=(data_size - length))
                    sample = slice_data[random_start: random_start + length]
                    test_sample.append(sample)
                # print(len(train_sample))
            train_samples[count] = train_sample
            test_samples[count] = test_sample
        return train_samples, test_samples

    # 打标签
    def add_label(train_test):
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

    # 用训练集标准差,标准化训练集和测试集
    def scalar_stand(Train_X, Test_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    # 测试集切分为 测试集和验证集
    def valid_test_slice(test_x, test_y):
        test_size = rate[2]/(rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for valid_index, test_index in ss.split(test_x, test_y):
            x_valid, x_test = test_x[valid_index], test_x[test_index]
            y_valid, y_test = test_y[valid_index], test_y[test_index]
            return x_valid, y_valid, x_test, y_test

    # 读数据,结果是一个三维的 [9][20][20w+] 九种状态 每种有20组数据 每组数据长度为20w+
    data = capture(d_path)
    train, test = slice_data(data)
    # print(len(train[0]), len(test))
    # 为数据打标签
    train_x, train_y = add_label(train)
    test_x, test_y = add_label(test)
    # 对结果进行 one-hot 编码
    train_y, test_y = one_hot(train_y, test_y)
    # 标准化 训练集和测试集
    train_x, test_x = scalar_stand(train_x, test_x)
    # 切分测试集为 验证集+测试集
    valid_x, valid_y, test_x, test_y = valid_test_slice(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

if __name__ == "__main__":
    filename = r'D:\Data\Paderborn\data'
    preprocess(filename)