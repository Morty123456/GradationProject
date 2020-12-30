import matplotlib.pyplot as plt
from scipy.io import loadmat


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


def draw(data):
    plt.plot(data)
    plt.ylabel("Grade")
    plt.ylabel("number")
    # plt.axis([-1, 11, 0, 7])
    plt.savefig('test', dpi=600)
    plt.show()


filename = r'D:\Data\Paderborn\data\KA01\N09_M07_F10_KA01_1.mat'
data = load_file(filename)
# print(data[0][1:2000])
# print(data[1][1:2000])
# print(data[0][1:100])
x = 0
draw(data[1][x:x+300])
# draw(data[1][x:x+2100])

