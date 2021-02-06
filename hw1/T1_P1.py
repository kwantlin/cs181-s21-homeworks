import numpy as np
import math
from sympy import symbols, diff

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 100

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])

N = len(data)

def compute_loss(W):
    ## TO DO
    loss = 0
    for n in range(N):
        numer = 0
        denom = 0
        for i in range(N):
            if i != n:
                diff = np.array([data[i][0] - data[n][0], data[i][1] - data[n][1]])
                diff.shape = (2,1)
                diff_T = diff.copy()
                diff_T.shape = (1,2)
                # print("Diff vectors", diff, diff.shape, diff_T, diff_T.shape)
                a = np.matmul(W, diff)
                b = np.ndarray.item(np.matmul(diff_T, a))
                b = math.exp(-b)
                denom += b
                b *= data[i][2]
                numer += b
        loss += (data[n][2] - numer/denom) ** 2
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))