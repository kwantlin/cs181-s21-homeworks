import numpy as np
import scipy
import matplotlib.pyplot as plt

import time

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        self.loss = []
        print(X)
        N = len(X)
        ones = np.ones((N,1))
        X = np.concatenate((np.ones((N,1)), X), axis=1)
        print(X)
        self.W = np.random.rand(3, 3) # each row is for a class; each column for a component of x
        print(self.W)
        # time.sleep(10)
        for j in range(200000):
            if j % 10000 == 0:
                    print(j)
            avg_grad = np.zeros((3,3)) # same dimensional scheme as W
            for k in range(3):
                multiplied = np.matmul(self.W, X.T)
                softm = scipy.special.softmax(multiplied, axis=0)
                yhat = softm[k]
                yactual = [1 if yterm == k else 0 for yterm in y]
                avg_grad[k] = np.dot((yhat-yactual).T, X)/N + self.lam*self.W[k]
            self.W = self.W - self.eta*avg_grad
            multiplied = np.matmul(self.W, X.T)
            softm = scipy.special.softmax(multiplied, axis=0)
            # print(softm.shape)
            logged = np.log(softm)
            # print("Logged shape", logged.shape)
            yactual = np.array([[1 if yterm == k else 0 for k in range(3)] for yterm in y])
            # print(yactual.shape)
            # print(len(yactual), len(yactual[0]))
            loss = np.matmul(yactual, logged)
            # print(-np.trace(loss))
            self.loss.append(-np.trace(loss))
        print("Logistic Loss: ", self.loss[-1])
        print(self.W)
        return self.W

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        print("Shape", X_pred.shape)
        X_pred = np.concatenate((np.ones((len(X_pred),1)), X_pred), axis=1)
        preds = []
        for x in X_pred:
            multiplied = np.matmul(self.W, x.T)
            # print(multiplied.shape)
            # print("Multiplied", multiplied)
            softm = scipy.special.softmax(multiplied, axis=0)
            # print("Soft m", softm)
            # print("Class from softm", np.argmax(softm))
            preds.append(np.argmax(softm))
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        iterations = list(range(1,len(self.loss)+1))
        loss = self.loss
        plt.plot(iterations, loss)
        plt.title('Loss Over Gradient Descent')
        plt.xlabel('Number of Iterations')
        plt.ylabel('NegativeLog-Likelihood  Loss')
        plt.savefig('LogRegLoss.png')
        plt.show()
