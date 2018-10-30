import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# create mock data
X, Y = make_blobs(n_samples=100, centers=2, random_state=6)
x1, x2, y1, y2 = [], [], [], []
for x, y in zip(X, Y):
	if y == 1:
		x1.append(x[0])
		y1.append(x[1])
	elif y == 0:
		x2.append(x[0])
		y2.append(x[1])

# create the support vector machine classifier 
clf = svm.SVC(kernel='linear')#, C=1000)

# train the classifier
clf.fit(X, Y)

# plot the data
plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='g')

# draw the linear classification lines
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