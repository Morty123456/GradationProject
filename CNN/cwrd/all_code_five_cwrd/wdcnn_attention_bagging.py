from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation, BatchNormalization, Input, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.layers.core import *
import preprocess
import numpy as np
import keras.backend as K


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


model = wdcnn_attention_model()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=20, validation_data=(x_valid, y_valid))
# 输出损失和精度

y_predict = model.predict(x_test)

count = 0
for i in range(len(y_test)):
    predict = y_predict[i]
    test = y_test[i]
    predict_local = 0
    test_local = 0
    for j in range(len(predict)):
        if predict[j] > predict[predict_local]:
            predict_local = j
        if test[j] > test[test_local]:
            test_local = j
    # print(i)
    # if i < 10:
    #     print(predict_local, test)
    if predict_local == test_local:
        count = count+1
print(count)


score = model.evaluate(x=x_test, y=y_test)
print(score[0])
print(score[1])