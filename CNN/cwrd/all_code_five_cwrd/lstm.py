from keras.layers import Dense, Flatten, LSTM
from keras.models import Sequential
from keras.utils import plot_model
import preprocess
from keras.callbacks import TensorBoard
import numpy as np
import time

# 训练参数
# batch_size = 128
# epochs = 3
# num_classes = 10
# length = 2048
# BatchNorm = True  # 是否批量归一化
# number = 1000  # 每类样本的数量
# normal = True  # 是否标准化
# rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例
date = time.strftime("%Y%m%d", time.localtime())
mark = time.strftime("%Y%m%d_%H%M", time.localtime())

path = r'../data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path, length=2048,
                                                                       number=1000,
                                                                       normal=True,
                                                                       rate=[0.7, 0.2, 0.1],
                                                                       enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]

input_shape = x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

model_name = "lstm_diagnosis-{}".format(mark)


def lstm_model():
    model = Sequential()
    model.add(LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = lstm_model()
# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=128, epochs=20,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True)

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=model, to_file='lstm-diagnosis.png', show_shapes=True)