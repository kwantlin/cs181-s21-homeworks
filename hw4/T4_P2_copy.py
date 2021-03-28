# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy", allow_pickle=False)

print(pics.shape)

small_dataset = np.load("data/small_dataset.npy")

small_labels = np.load("data/small_dataset_labels.npy").astype(int)

large_dataset = np.load("data/large_dataset.npy")

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.losses = []

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        N = len(X)
        while True:
            protos = np.random.randn(self.K,N,28,28)
            rs = np.zeros((N, self.K))

            # get indicator matrix
            for n in range(N):
                if n % 500 == 0:
                    print(n)
                xn = X[n]
                diffs = np.zeros(self.K)
                for k in range(self.K):
                    diffs[k] = np.linalg.norm(protos[k] - xn)
                best_k = np.argmin(diffs)
                rs[n][best_k] = 1
            print("rs done")
            
            # calculate the loss
            L = 0
            for n in range(N):
                for k in range(self.K):
                    L += rs[n][k] * (np.dot((X[n] - protos[k]), (X[n] - protos[k])))
            self.losses.append(L)
            print("got loss")

            if (self.losses[-2] - self.losses[-1])/(self.losses[-2]) < 0.01:
                break

            # update prototypes 
            for k in range(self.K):
                numer = 0
                denom = 0
                for n in range(N):
                    numer += rs[n][k]*X[n]
                    denom += rs[n][k]
                protos[k] = numer / denom
        

        

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        pass

K = 10
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(pics)
plt.plot(KMeansClassifier.losses, linestyle = 'dotted')
plot.savefig("kmeans-loss.png")
plt.show()

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(pics[0].reshape(28,28), cmap='Greys_r')
plt.show()


class HAC(object):
	def __init__(self, linkage):
		self.linkage = linkage
