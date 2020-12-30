import os


# 九个文件分别是:
# K001 data0: 健康
# KA01 data1: 人为-外圈-电火花加工
# KA03 data2: 人为-外圈-电动雕刻机
# KA04 data3: 寿命-外圈-疲劳
# KA07 data4: 人为-外圈-钻探
# KA15 data5: 寿命-外圈-塑性变形
# KI01 data6: 人为-内圈-电火花加工
# KI03 data7: 人为-内圈-电动雕刻机
# KI04 data8: 寿命-内圈-疲劳
# 文件路径 数据类型(故障种类)
def load_data(file_path, type):
    f = open(file_path)
    res = []
    while 1:
        line = f.readline()
        if not line:
            break
        dataStr = line.split(" ")
        data = []
        for i in range(len(dataStr)-1):
            data.append(float(dataStr[i]))
        data.append(type)
        if len(data) > 100:
            res.append(data)
    return res


if __name__ == "__main__":
    file_path = r'D:\WorkSpace\PyCharm\GradationProject\Paderborn\NineCtaegories\prodata'
    files = os.listdir(file_path)
    for file in files:
        file = os.path.join(file_path, file)
        data = load_data(file, 0)
        print(data[0])
        print(len(data), len(data[0]))
        break
