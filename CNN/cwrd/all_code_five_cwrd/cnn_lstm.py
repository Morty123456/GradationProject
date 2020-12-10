from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
import preprocess
import numpy as np

path = r'../data/0HP'
# 得到训练集、验证集、测试集 的数据
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path, length=2048,
                                                                       number=1000,
                                                                       normal=True,
                                                                       rate=[0.7, 0.2, 0.1],
                                                                       enc=True, enc_step=28)
# 将训练集、验证集、测试集 增加一个维度
x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]

# 得到输入的维度
input_shape = x_train.shape[1:]


def lstm_classification():
    model = Sequential()
    # 在lstm之前加上一层卷积来提取特征，注意要最大池化，才可以提升准确率
    model.add(Conv1D(filters=16, kernel_size=64, strides=8, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(MaxPooling1D(4))
    # units 门结构使用的隐藏单元个数
    model.add(LSTM(units=32, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True))
    # 从lstm到全连接需要展平
    model.add(Flatten())
    # 添加全连接层，分类目标为10个
    model.add(Dense(units=10, activation='softmax'))
    # 编译模型
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = lstm_classification()
# 执行训练过程
model.fit(x=x_train, y=y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_valid, y_valid), shuffle=True)
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率:", score[0])
print("测试集上的准确率:", score[1])
plot_model(model=model, to_file='lstm-diagnosis.png', show_shapes=True)