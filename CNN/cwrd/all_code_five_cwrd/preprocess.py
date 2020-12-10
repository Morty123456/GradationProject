from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

# HG数据解析
# 这里面的数据是 九种故障+正常状态=十种状态 下的时间序列数据
# 每一种数据都有 12w-20w 个
#

def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                # 找到 DE 驱动端 加速度信息，添加到 files 中
                if 'DE' in key:
                    files[i] = file[key].ravel()
        # print("故障的类型个数：")
        # print(len(files))
        # print("每类故障的数据个数：")
        # for f in files:
        #     print(len(files[f]))
        # 这里得到的是 每类故障的振动信息
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            # all_length：所有数据的长度
            all_lenght = len(slice_data)
            # 切分训练集 和 测试集/验证集
            # 训练集数据的范围（也就是训练集数据的下标都要在end_index之前）
            end_index = int(all_lenght * (1 - slice_rate))
            # 训练集的样本数量
            samp_train = int(number * (1 - slice_rate))  # 500
            Train_sample = []
            Test_Sample = []
            # 训练集是否需要数据增强
            if enc:
                # 这是以 enc_step 为步长进行数据增强，每次可以取样的数量
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    # 训练集的取数据的随机开始地址的范围是 0 —— 训练集大小-2*取样长度
                    # 这里减两个取样长度，是因为要采用数据增强
                    # 数据增强的一个循环内，最后一次的取样末尾 - 第一次取样的开始 = 2*取样长度
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    # 通过在这里循环，来保证有很多数据 在 取样长度内有重叠
                    # 在生成一个随机起始地址后，要以此地址循环enc_time次，每次起始地址都是上一次的加 enc_step
                    # 这样就可以保证这 enc_time 次取样，都有互相重叠的部分
                    for h in range(enc_time):
                        samp_step += 1
                        # 训练集取样本的真正 随机开始地址
                        random_start += enc_step
                        # 从此地址向后取 取样长度大小的数据
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        # 如果样本数达到要求，就退出
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    # length是取样长度，是两个周期的大小 864
                    # 训练集样本数据的起始地址是 随机取样的，范围是 0 —— 训练集大小-取样长度
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    # 从随机取样的起始地址开始，向后取 length，作为一个训练集样本
                    sample = slice_data[random_start:random_start + length]
                    # 把这样本，填到此类别的数据组中
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                # 测试集的起始地址是随机取样，范围是 测试集大小 —— 数据末尾-取样长度
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                # 测试集和验证集样本，从随机取样的起始地址开始，向后取 length 大小，也就是两个周期大小
                sample = slice_data[random_start:random_start + length]
                # 样本添加到测试集中
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        # print("故障类型")
        # print(len(Train_Samples))
        # print(len(Test_Samples))
        # 返回得到的测试集和验证集样本
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        # print("打标签之后的数据")
        # print(len(X))
        # print(len(Y))
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        # 输入进来的 Train_Y[i] 还是单独的一个数字
        # np.array 就是为了把数字变为一维数组
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        # Encoder 把一维数组扩展为十维数组，第n位为1代表这是第n类数据(但这个时候还是小数)
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        # dtype 是把小数变为正数. 这时得到的结果就是整数了
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    # HG：取出每种故障类型的数据， K-V形式的数据，K是每个文件的名字，V是文件内的数据(数组形式的)
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    # 构造训练集、测试集和验证集
    # 训练集:包括十中状态,每种状态有500组数据,每组数据的长度是864（自己的理解是，维度是864维的）
    train, test = slice_enc(data)
    # print("测量数据的长度")
    # print(len(train))
    # for i in filenames:
    #     print(len(train[i]))
    #     print(len(train[i][0]))
    # 为训练集制作标签,返回X Y
    # HG:为训练集和测试集制作标签,编号为 0-9
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    # 计算训练集的平均值和标准差，以便测试集使用相同的变换，通过删除平均值和缩放到单位方差来标准化特征。
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    # 训练集 测试集 验证集 的数目分别为 10*500*864 10*250*864 10*250*864
    # 轴承有十种状态，每种状态有500组数据，每组数据有864个维度（维度是取得观测周期的两倍）
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    path = r'data\0HP'
    # length：是采样的长度
    # number：是采样的个数（训练集+验证集+测试集）
    # normal：是否标准化
    # rate：各个数据集所占的比例
    # enc：是否进行数据增强
    # enc_step：数据增强顺延间隔
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=False,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)