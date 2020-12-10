from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

x, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
# 决策树
clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores1 = cross_val_score(clf1, x, y)
print(scores1)
# 随机森林
clf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores2 = cross_val_score(clf2, x, y)
print(scores2)
# ExtraTree分类器集合
clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores3 = cross_val_score(clf3, x, y)
print(scores3)