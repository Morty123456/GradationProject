import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from tensorflow import itertools

CSV_PATH = './data_humidity.csv'
df = pd.read_csv(CSV_PATH)
# 分割数据和标签。X为输入，是245维的数据。 Y是输出，是1维的数据
X = np.expand_dims(df.values[:, 0:246].astype(float), axis=2)
Y = df.values[:, 246]

# 这部分没咋看懂，以后再说
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
# print(Y_onehot)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)

def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(246, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)  # 保存网络结构为图片，这一步可以直接去掉
    print(model.summary())  # 显示网络结构
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=40, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train)