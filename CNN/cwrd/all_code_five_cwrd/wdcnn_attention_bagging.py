from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation, BatchNormalization, Input, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.layers.core import *
import preprocess
import numpy as np
import keras.backend as K
import random


# 训练参数
batch_size = 128   # 一次训练选取的样本数
epochs = 20        # 训练轮数
num_classes = 10   # 数据类别
length = 2048      # 每组数据长度
BatchNorm = True   # 是否批量化归一
number = 1000      # 每类样本的数量
normal = True      # 是否标准化
rate = [0.7, 0.2, 0.1]   # 训练 验证 测试 划分
path = r'../data/0HP'


x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path, length=length, number=number, normal=normal, rate=rate, enc=True, enc_step=28)

# np.newaxis 增加一个维度，把二维转化为三维
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]

# 输入的是2维的
input_shape = x_train.shape[1:]


def attention_3d_block(inputs, single_attention_vector=False):
    # print(inputs)
    # 返回输入的元组，以列表的形式存储
    time_steps = K.int_shape(inputs)[1]
    # print(time_steps)
    # 输入是 [训练批量 时间长度 输入维度]
    input_dim = K.int_shape(inputs)[2]
    # print(input_dim)
    # Permute层, 对样本模式进行重排
    # (2, 1) 代表将输入的第二个维度重排到输出的第一个维度，输入的第一个维度重排到输出的第二个维度
    # print(inputs)
    a = Permute((2, 1))(inputs)
    # print(a)
    # Dense全连接层
    a = Dense(time_steps, activation='softmax')(a)
    # print(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    # print(a_probs)
    # Multiply 计算输入张量列表的乘积。
    # 输入：张量列表，张量具有相同的尺寸
    # 输出：一个张量，尺寸和输入的相同
    output_attention_mul = Multiply()([inputs, a_probs])
    # print(output_attention_mul)
    return output_attention_mul

# 定义卷积层
def wdcnn(model, filters, kernerl_size, strides, conv_padding, pool_padding, pool_size, BatchNormal):
    model = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides, padding=conv_padding, kernel_regularizer=l2(1e-4))(model)
    if BatchNormal:
        model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=pool_size, padding=pool_padding)(model)
    return model


def wdcnn_attention_model():
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = wdcnn(x, filters=32, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
              BatchNormal=True)
    x = wdcnn(x, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
              BatchNormal=True)
    x = wdcnn(x, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
              BatchNormal=True)
    x = wdcnn(x, filters=64, kernerl_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
              BatchNormal=True)
    x = attention_3d_block(x)
    x = Flatten()(x)
    x = Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4))(x)
    output = Dense(units=10, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=[inputs], outputs=output)
    return model


# 得到训练集, 每组训练集占全部训练集的比例是 rate
def get_train(x_train, y_train, rate):
    if rate == 1:
        return x_train, y_train
    x = []
    y = []
    count = 0
    length = len(x_train)
    while count < length*rate:
        local = random.randint(0, length-1)
        x.append(x_train[local])
        y.append(y_train[local])
        count = count+1
    # list 变为 array
    x, y = np.asarray(x), np.asarray(y)
    return x, y


# 从数组中得到是类别
def get_class(predict):
    res = []
    for p in predict:
        local = 0
        for i in range(len(p)):
            if p[i] > p[local]:
                local = i
        res.append(local)
    return res

# 训练函数
def get_modelPredict(rate):
    model = wdcnn_attention_model()
    x, y = get_train(x_train, y_train, rate)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=batch_size, epochs=20, validation_data=(x_valid, y_valid))
    # model.predict 返回的是一个长度为10的数组
    y_predict = model.predict(x_test)
    return y_predict

# bagging 投票进行分类选择
def bagging():
    predict1 = get_modelPredict(1)
    predict2 = get_modelPredict(0.8)
    predict3 = get_modelPredict(0.8)
    predict4 = get_modelPredict(0.8)
    predict5 = get_modelPredict(0.8)
    res_bagging = []
    for i in range(len(predict1)):
        res_line = []
        for j in range(len(predict1[0])):
            pre = predict1[i][j]+(predict2[i][j]+predict3[i][j]+predict4[i][j]+predict5[i][j])*0.8
            res_line.append(pre)
        res_bagging.append(res_line)
    class_predict = get_class(res_bagging)
    return class_predict


predict = bagging()
test = get_class(y_test)
count = 0
for i in range(len(predict)):
    if predict[i] == test[i]:
        count = count+1
print(count/len(predict))

# test = []
# for i in range(5):
#     test.append(i)
# print(test[0])