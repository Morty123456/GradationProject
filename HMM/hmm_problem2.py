import numpy as np
from hmmlearn import hmm

states = ["box1", "box2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0], [1], [0], [1], [0], [0], [0], [1], [1], [0], [1], [1]])

model2.fit(X2, lengths=[4, 4, 4])
# model2.fit(X2)

# 初始化概率
print(model2.startprob_)
# 隐藏状态转移概率
print(model2.transmat_)
# 隐藏-观测 的 混淆矩阵
print(model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))