from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model

from keras.layers.core import *
import preprocess
import keras.backend as K


# 加入注意力机制
def attention_3d_block(inputs, single_attention_vector=False):
    # print(inputs)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    # Permute层, 对样本模式进行重排
    # (2, 1) 代表将输入的第二个维度重排到输出的第一个维度，输入的第一个维度重排到输出的第二个维度
    a = Permute((2, 1))(inputs)
    # Dense全连接层
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def attention_model():
    inputs = Input(shape=(20, 7))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
    # dropout: 防止过拟合,神经元以0.3的概率停止工作
    x = Dropout(0.3)(x)
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


path = '../pollution.csv'
train_X, train_Y = preprocess.prepro(path)
model = attention_model()
model.compile(optimizer='adam', loss='mse')
model.fit([train_X], train_Y, epochs=5, batch_size=64, validation_split=0.1)
plot_model(model=model, to_file='attentionLSTM.png', show_shapes=True)