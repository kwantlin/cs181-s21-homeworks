import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

x1 = np.array([-3, -2, -1, 0, 1, 2, 3])
x2 = (-8/3)*(x1**2) + (2/3)*(x1**4)
plt.scatter(x1, x2)
plt.savefig("training-plot.png")
plt.show()

X = np.zeros((len(x1), 2))
for i in range(len(x1)):
    X[i][0] = x1[i]
    X[i][1] = x2[i]
y = np.array([1, 1, -1, 1, -1, 1, 1])
print(X)
print(y)
# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.xlabel('phi1')
plt.ylabel('phi2')
plt.savefig("svm.png")
plt.show()