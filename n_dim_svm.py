import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sys

# create mock data
X, Y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=6)
x1, x2, y1, y2 = [], [], [], []
for x, y in zip(X, Y):
	if y == 1:
		x1.append(x[0])
		y1.append(x[1])
	elif y == 0:
		x2.append(x[0])
		y2.append(x[1])

# create the support vector machine classifier
clf = SVC()

# train the classifier
clf.fit(X, Y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# plot the data
plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='g')

# draw the classification lines
ax = plt.gca()
x_range = ax.get_xlim()
y_range = ax.get_ylim()
y, x = np.meshgrid(
	np.linspace(y_range[0], y_range[1], 40),
	np.linspace(x_range[0], x_range[1], 40))
xy = np.vstack([x.ravel(), y.ravel()]).T
z = clf.decision_function(xy).reshape(x.shape)
ax.contour(x, y, z, colors='black', levels=[-1, 0, 1],
	linestyles=['dotted', 'solid', 'dotted'])
plt.show()
