import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime

data = pd.read_csv("./Iris.csv")
# print(data)
# x作为特征状态，y作为标签
x1 = data['SepalLengthCm']
x2 = data['SepalWidthCm']
x3 = data['PetalLengthCm']
x4 = data['PetalWidthCm']
y = data['Species']
# print(y)
x_setosa = [[0 for col in range(4)] for row in range(50)]
x_versicolor = [[0 for col in range(4)] for row in range(50)]
x_virginica = [[0 for col in range(4)] for row in range(50)]
y_setosa = {}
y_versicolor = {}
y_virginica = {}
for i in range(0, 50):
    x_setosa[i][0] = x1[i]
    x_setosa[i][1] = x2[i]
    x_setosa[i][2] = x3[i]
    x_setosa[i][3] = x4[i]
    y_setosa[i] = y[i]
for i in range(50, 100):
    x_setosa[i-50][0] = x1[i]
    x_setosa[i-50][1] = x2[i]
    x_setosa[i-50][2] = x3[i]
    x_setosa[i-50][3] = x4[i]
    y_setosa[i-50] = y[i]
for i in range(100, 150):
    x_setosa[i-100][0] = x1[i]
    x_setosa[i-100][1] = x2[i]
    x_setosa[i-100][2] = x3[i]
    x_setosa[i-100][3] = x4[i]
    y_setosa[i-100] = y[i]
# print(y_setosa)
A = np.column_stack([x1, x2, x3, x4])
# print(A)
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=2000).fit(A)
hidden_states = model.predict(A)
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (hidden_states == i)
    plt.legend()
plt.show()