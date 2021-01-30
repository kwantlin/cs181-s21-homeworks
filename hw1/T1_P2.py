#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

from operator import itemgetter 

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']
W1 = np.array([[1., 0.], [0., 1.]])

X = X_df.values
y = y_df.values

N = X.shape[0]

print("y is:")
print(y)

def mahalanobis(a, b, W, alpha=1):
    diff = np.array([a[0] - b[0], a[1] - b[1]])
    diff.shape = (2,1)
    diff_T = diff.copy()
    diff_T.shape = (1,2)
    # print("Diff vectors", diff, diff.shape, diff_T, diff_T.shape)
    W = alpha * W
    a = np.matmul(W, diff)
    b = np.ndarray.item(np.matmul(diff_T, a))
    b = math.exp(-b)
    return b

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    # TODO: your code here
    y_df = []
    for n in range(N):
        numer = 0
        denom = 0
        for i in range(N):
            if i != n:
                d = mahalanobis(X[i], X[n], W1, alpha)
                denom += d
                d *= y[i]
                numer += d
        y_df.append(numer/denom)
    return y_df

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    y_df = []
    for n in range(N):
        dist = {}
        for i in range(N):
            if i != n:
                dist[i] = mahalanobis(X[i], X[n], W1)
        res = dict(sorted(dist.items(), key = itemgetter(1))[-k:]) 
        # print("Indices for k=", k, res)
        y_vals = [y[i] for i in res.keys()]
        x_vals = [(X[i][0], X[i][1]) for i in res.keys()]
        # print("Summary: k=", k, "x=", X[n], "xvals:", x_vals,"yvals:", y_vals)
        y_df.append(sum(y_vals)/k)
    return y_df

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print("y_pred with alpha ", alpha, y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print("y_pred with k ", k, y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
