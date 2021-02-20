import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        print(X)
        N = len(X)
        m = len(X[0])
        self.pi = np.zeros(3)
        ylist = list(y)
        for k in range(3):
            self.pi[k] = ylist.count(k) / N
        print("PI's", self.pi)

        self.mu = np.zeros((2,3))
        for k in range(3):
            numer = 0
            for i in range(N):
                if y[i] == k:
                    numer += X[i]
            self.mu[:, k] = numer/ylist.count(k)
        print("Mu's", self.mu)
        print(self.mu[:, 2].shape)

        if self.is_shared_covariance:
            self.cov = np.zeros((m,m))
            numer = np.zeros((m,m))
            for i in range(N):
                for k in range(3):
                    if y[i] == k:
                        vec = X[i].T - self.mu[:, k]
                        # print(vec)
                        numer = numer + np.outer(vec.ravel(), vec.T.ravel()) 
                        # print(numer)
            self.cov = numer / N
            # self.cov = np.cov(X.T)
            # print(np.cov(X.T))
            print("Single cov \n", self.cov)
            # np.cov on all data points
        else:
            self.covs = []
            for k in range(3):
                numer = np.zeros((m,m))
                for i in range(N):
                    if y[i] ==k:
                        vec = X[i].T - self.mu[:, k]
                        vec.reshape(vec.shape[0],-1)
                        # print(vec)
                        numer = numer + np.outer(vec.ravel(), vec.T.ravel()) 
                        print(numer)
                self.covs.append(numer / ylist.count(k))
                # filter_arr = y % 3 == k
                # newarr = X[filter_arr]
                # # print(newarr)
                # # print("NP COV: ", np.cov(newarr.T))
                # self.covs.append(np.cov(newarr.T))
            print("Multiple covs \n", self.covs)
                #call np.cov on all data points for each class
        loss = 0
        for i in range(N):
            for k in range(3):
                if y[i] == k:
                    if self.is_shared_covariance:
                        loss -= np.log(mvn(self.mu[:, k], self.cov).pdf(X[i].T))
                    else:
                        loss -= np.log(mvn(self.mu[:, k], self.covs[k]).pdf(X[i].T))
                    loss -= np.log(self.pi[k])
        print("Gaussian Loss: ", loss)

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            if self.is_shared_covariance:
                class_preds = np.zeros(3)
                for k in range(3):
                    val = mvn(self.mu[:, k], self.cov).pdf(x.T)
                    val *= self.pi[k]
                    class_preds[k] = val
                # print(class_preds)
                preds.append(np.argmax(class_preds))
                # print(preds[-1])
            else:
                class_preds = np.zeros(3)
                for k in range(3):
                    val = mvn(self.mu[:, k], self.covs[k]).pdf(x.T)
                    val *= self.pi[k]
                    class_preds[k] = val
                # print(class_preds)
                preds.append(np.argmax(class_preds))
                # print(preds[-1])
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        pass
