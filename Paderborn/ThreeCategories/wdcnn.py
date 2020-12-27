from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
import Paderborn.ThreeCategories.preprocess as preprocess
import numpy as np

# 训练参数
# 训练保持在100轮; 每组数据最好包括一个周期,大概1800个数据
batch_size = 128  # 一次训练选取的样本数
epochs = 100       # 训练轮数
num_classes = 3   # 数据类别
length = 1800    # 每组数据长度 1800效果比较好
number = 200     # 每类样本的数量
rate = [0.7, 0.2, 0.1]  # 训练 验证 测试 划分
path = r'D:\Data\Paderborn\data'

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path, length=length, number=number, rate=rate
)
# print(len(x_train), len(y_train), len(x_valid), len(y_valid), len(x_test), len(y_test))

# np.newaxis 增加一个维度，把二维转化为三维
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
# 输入是二维的
input_shape = x_train.shape[1:]

# 定义卷积层
def wdcnn(model, filters, kernerl_size, strides, conv_padding, pool_padding, pool_size, BatchNormal):
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides, padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
    return model


def wdcnn_model():
    # 实例化序贯模型
    model = Sequential()
    # 第一层 宽卷积层
    model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 深度卷积，2-5层
    model = wdcnn(model, filters=32, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=True)
    model = wdcnn(model, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=True)
    model = wdcnn(model, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=True)
    model = wdcnn(model, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=True)
    # 从卷积到全连接需要展平
    model.add(Flatten())
    # 添加全连接层
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
    # 增加输出层
    model.add(Dense(units=3, activation='softmax', kernel_regularizer=l2(1e-4)))
    # plot_model(model, to_file='./wdcnn_hgd.png', show_shapes=True)
    # 编译模型，添加评价函数和损失函数，不过评价函数和损失函数不会用于训练过程中
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = wdcnn_model()
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=100, validation_data=(x_valid, y_valid))
y_predict = model.predict(x_test)

# 统计预测准确的个数
total = len(y_predict)
count = 0
for i in range(total):
    length = len(y_predict[i])
    local_predict = 0
    local_test = 0
    for j in range(length):
        if y_predict[i][j] > y_predict[i][local_predict]:
            local_predict = j
        if y_test[i][j] > y_test[i][local_test]:
            local_test = j
    if local_test == local_predict:
        count += 1

print(count/total)
# print(model.predict(x_test))
# print(y_test)
# 输出损失和精度
# score = model.evaluate(x=x_test, y=y_test)
# print(score[0])
# print(score[1])