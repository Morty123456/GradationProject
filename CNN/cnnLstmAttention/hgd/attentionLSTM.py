from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model

from keras.layers.core import *
import preprocess
import keras.backend as K


# 加入注意力机制
# 将输入来一个全连接，然后使用softmax函数来计算概率
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


def attention_model():
    inputs = Input(shape=(20, 7))
    # 输出为 20*64
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same',必要时使用0进行填充
    # dropout: 防止过拟合,神经元以0.3的概率停止工作
    x = Dropout(0.3)(x)
    # 输出依旧为 20*64
    # Bidirectional RNN 双向循环网络
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


path = '../pollution.csv'
rate = [0.7, 0.2, 0.1]
# 利用前20个时间点的 7维的数据，去预测下一个时间点的 某个数据（这个数据也在七维之中）
train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess.prepro(path, rate)
print(train_x.shape, train_y.shape)

model = attention_model()
model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=5, batch_size=64, validation_data=[valid_x, valid_y])
# evaluate输入数据和标签，输出损失和精度（只能分类，不能预测）
# score = model.evaluate(test_x, test_y)
# print(score[0])
# print(score[1])
# predict输入测试数据，输出预测结果
predict = model.predict(test_x)
loss = 0
for i in range(len(predict)):
    loss = loss + (predict[i]-test_y[i])*(predict[i]-test_y[i])
print(loss)
# print(len(predict), len(test_y))
plot_model(model=model, to_file='attentionLSTM.png', show_shapes=True)