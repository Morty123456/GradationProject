import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score, train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

CSV_PATH = './data_humidity.csv'
df = pd.read_csv(CSV_PATH)
# 分割数据和标签。X为输入，是246维的数据。 Y是输出，是1维的数据
X = np.expand_dims(df.values[:, 0:246].astype(float), axis=2)
Y = df.values[:, 246]

# 这部分没咋看懂，以后再说
encoder = LabelEncoder()
# 变为从0开始的编码
Y_encoded = encoder.fit_transform(Y)
# 变为n为编码
Y_onehot = np_utils.to_categorical(Y_encoded)
# print(Y_onehot)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)
print(len(X_train))
def baseline_model():
    model = Sequential()
    # 使用六层Conv1D来提取特征，每两层后添加一层MaxPooling1D来保留主要特征
    # 16个卷积核, 宽度为3, 输入维度为246
    # 卷积之后变为 244*16
    model.add(Conv1D(16, 3, input_shape=(246, 1)))
    # 卷积之后变为 242*16
    model.add(Conv1D(16, 3, activation='tanh'))
    # 最大池化,维度变为之前的1/3 80*16
    model.add(MaxPooling1D(3))
    # 卷积之后变为 78*64
    model.add(Conv1D(64, 3, activation='tanh'))
    # 卷积之后变为 76*64
    model.add(Conv1D(64, 3, activation='tanh'))
    # 最大池化 维度变为 25*64
    model.add(MaxPooling1D(3))
    # 卷积之后变为 23*64
    model.add(Conv1D(64, 3, activation='tanh'))
    # 卷积之后变为 21*64
    model.add(Conv1D(64, 3, activation='tanh'))
    # 最大池化 维度变为 7*64
    model.add(MaxPooling1D(3))
    # 压平，变为一维数据 448
    model.add(Flatten())
    # 全连接层
    model.add(Dense(9, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)  # 保存网络结构为图片，这一步可以直接去掉
    print(model.summary())  # 显示网络结构
    # 交叉熵损失函数作为模型训练的损失函数
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=4, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train)

# 混淆矩阵定义
def plot_confusion_matrix(cm, classes, titile='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    plt.yticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))

# 将模型转化为json
model_json = estimator.model.to_json()
with open(r"./model.json", 'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
# 网络的参数保存在 model.h5 中
estimator.model.save_weights('model.h5')


# 加载模型用做预测
json_file = open(r"./model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# 加载网络结构
loaded_model = model_from_json(loaded_model_json)
# 加载网络参数
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别
predicted = loaded_model.predict(X)
predicted_label = loaded_model.predict_classes(X)
print("predicted label:\n " + str(predicted_label))
# 显示混淆矩阵
# plot_confuse(estimator.model, X_test, Y_test)