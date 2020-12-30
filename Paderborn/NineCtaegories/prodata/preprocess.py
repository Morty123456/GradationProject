import os
from scipy.io import loadmat
import numpy as np
from numpy import *
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

# 文件路径, 每组数据长度, 数据大小, 训练验证测试比例
def preprocess(d_path, length=1800, number=1000):

    # 加载此路径下的文件
    # 总共有九种类别
    # 每组类别下都只读取N09_M07_F10测试条件下的数据
    def capture(path):
        folders = os.listdir(path)
        data = {}
        count = 0
        print(folders)
        # 九种数据
        for folder in folders:
            folder_path = os.path.join(path, folder)
            phrase_data = capture_folder(folder_path)
            data[count] = phrase_data
            count += 1
        # print(len(data))
        return data

    # 读取此文件夹中的数据（一个文件夹下的20个文件）
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
                # print(len(data[0]))
                # print(data[0])
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
    def slice_data(data):
        samples = {}
        # 每种类型的数据取number组,每种类型的数据 有len(data[0])个文件，所以每个文件取ratenumber个数据
        ratenumber = int(number/len(data[0]))
        # 九种类别的数据
        for count in range(len(data)):
            train_sample = []
            # 每种数据有20个文件
            for thisCount in range(len(data[count])):
                # 本次要切割的数据
                slice_data = data[count][thisCount]
                data_size = len(slice_data)
                # 选择训练集的数据
                # 从某个随机位置开始,向后取length长度的数据
                for i in range(ratenumber):
                    random_start = np.random.randint(low=0, high=(data_size - length))
                    sample = slice_data[random_start: random_start + length]
                    train_sample.append(sample)
                # print(len(train_sample))
            samples[count] = train_sample
        return samples

    # 写数据，把处理过的数据写入 文件中
    def write_file(data, filepath):
        # 九种故障状态
        for i in range(len(data)):
            dataState = data[i]
            filename = '\data' + str(i) + '.txt'
            f = open(filepath+filename, 'a')
            for j in range(len(data[i])):
                writeInData = data[i][j]
                ins = ""
                for num in writeInData:
                    ins = ins + str(num) + " "
                f.write(ins+'\n')
                print(len(writeInData))


    # 读数据,结果是一个三维的 [9][20][20w+] 九种状态 每种有20组数据 每组数据长度为20w+
    data = capture(d_path)
    data = slice_data(data)
    print(len(data), len(data[0]), len(data[0][0]))
    filepath = r'D:\WorkSpace\PyCharm\GradationProject\Paderborn\NineCtaegories\prodata'
    # 数据写文件
    write_file(data, filepath)
    return data

if __name__ == "__main__":
    # 取消显示数组长度限制
    np.set_printoptions(threshold=np.inf)
    # 取消科学计数
    np.set_printoptions(suppress=True)

    filename = r'D:\Data\Paderborn\data'
    data = preprocess(filename)