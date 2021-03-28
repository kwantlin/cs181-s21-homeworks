# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time
from seaborn import heatmap

# This line loads the images for you. Don't change it!
# pics = np.load("data/images.npy", allow_pickle=False)

# print(pics.shape)

small_dataset = np.load("data/small_dataset.npy")

small_labels = np.load("data/small_dataset_labels.npy").astype(int)
# print(small_labels)

large_dataset = np.load("data/large_dataset.npy")

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
	# K is the K in KMeans
	def __init__(self, K):
		self.K = K
		self.losses = []

	def standard(self, X):
		means = np.mean(X, axis = 0)
		stdevs = np.std(X, axis = 0)
		stdevs = np.where(stdevs == 0, 1, stdevs)
		X = X - means
		X = X / stdevs
		# for i in range(X.shape[0]):
		# 	X[i] = np.subtract(X[i], means)
		# 	X[i] = np.divide(X[i], stdevs)
		return X
		# mean = np.mean(X)
		# stdev = np.std(X)
		# if stdev == 0:
		# 	stdev = 1
		# X = X - mean
		# X = X / stdev
		# return X

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X, standardize=False, get_cluster_ims=False):
		if standardize:
			X = self.standard(X.copy())
		N = X.shape[0]
		# x = X.reshape(*X.shape[:1], -1)
		# print(x.shape)
		self.protos = np.random.randn(self.K,784)
		rs = np.zeros(N)
		while True:
			diffs = np.zeros((N, self.K))
			for k in range(self.K):
				# print(np.linalg.norm(x - protos[k], axis=1))
				diffs[:, k] = np.linalg.norm(X - self.protos[k], axis=1)
			rs = np.argmin(diffs, axis=1)
			# print("rs done")
			
			for k in range(self.K):
				if k in rs:
					self.protos[k] = np.mean(X[rs == k], axis=0)

			L = 0
			for n in range(N):
				for k in range(self.K):
					if rs[n] == k:  
						L += (np.linalg.norm(X[n]- self.protos[k]))**2
			# print(L)
			self.losses.append(L)
			# print("got loss")
			if len(self.losses)>2 and self.losses[-1] == self.losses[-2]:
				if get_cluster_ims:
					self.cluster_images = []
					for k in range(self.K):
						self.cluster_images.append(X[rs == k])
				break
			if len(self.losses)>2 and self.losses[-1] > self.losses[-2]:
				print("ERROR")
				return
		

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.protos
		

np.random.seed(2)

# Question 2.1
K = 10
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(large_dataset)
plt.plot(KMeansClassifier.losses, linestyle = 'dotted')
plt.title("Objective Function (K=" + str(KMeansClassifier.K) + ")")
plt.savefig("2-1-losses.png")
plt.show()

# Check for correctness

check_dataset = np.load("P2_Autograder_Data.npy")
w = 10
h = 10
fig = plt.figure(figsize=(7, 9))
columns = 5
rows = 10

# ax enables access to manipulate each of subplots
ax = []
protos = np.zeros((5, 10, 784))
for i in range(5):
    KMeansClassifier = KMeans(K=10)
    KMeansClassifier.fit(check_dataset)
    protos[i] = KMeansClassifier.get_mean_images()

index = 0
for i in range(columns*rows):
        row    = (index // columns)
        col= index % columns
        img = protos[col][row].reshape(28,28)
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, index+1) )
        if index < 5:
            ax[-1].set_title("Restart #" + str(index))  # set title
        ax[-1].axis('off')
        if col == 0:
            ax[-1].text(-6, row+5, row)
        plt.imshow(img, cmap='Greys_r')
        index += 1
        # if c == 0:
        #     ax[-1].ylabel("Cluster #" + str(r))
plt.savefig("2-1-autogradercheck.png")
plt.show()  # finally, render the plot


# Question 2.2
losses = np.zeros((3, 5))
ks = [5, 10, 20]
for i in range(len(ks)):
    print("k =", ks[i])
    lossk = np.zeros((1,5))
    for j in range(5):
        KMeansClassifier = KMeans(K=ks[i])
        KMeansClassifier.fit(large_dataset)
        lossk[0][j] = KMeansClassifier.losses[-1]
        print(lossk)
    losses[i] = lossk
    print(losses)

y = np.mean(losses, axis = 1)
yerr = np.std(losses, axis = 1)
print(y)
print(yerr)

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(ks, y, yerr=yerr)
plt.title("Objective Function Value vs. K")
plt.savefig("2-2-objs.png")
plt.show()

# Question 2.3

w = 10
h = 10
fig = plt.figure(figsize=(7, 9))
columns = 3
rows = 10

# ax enables access to manipulate each of subplots
ax = []
protos = np.zeros((3, 10, 784))
for i in range(3):
    KMeansClassifier = KMeans(K=10)
    KMeansClassifier.fit(large_dataset)
    protos[i] = KMeansClassifier.get_mean_images()

index = 0
for i in range(columns*rows):
        row    = (index // columns)
        col= index % columns
        img = protos[col][row].reshape(28,28)
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, index+1) )
        if index < 3:
            ax[-1].set_title("Restart #" + str(index))  # set title
        ax[-1].axis('off')
        if col == 0:
            ax[-1].text(-6, row+5, row)
        plt.imshow(img, cmap='Greys_r')
        index += 1
        # if c == 0:
        #     ax[-1].ylabel("Cluster #" + str(r))

plt.savefig("2-3-meanimages.png")
plt.show()  # finally, render the plot



# Question 2.4

w = 10
h = 10
fig = plt.figure(figsize=(7, 9))
columns = 3
rows = 10

# ax enables access to manipulate each of subplots
ax = []
protos = np.zeros((3, 10, 784))
for i in range(3):
    KMeansClassifier = KMeans(K=10)
    KMeansClassifier.fit(large_dataset, True)
    protos[i] = KMeansClassifier.get_mean_images()

index = 0
for i in range(columns*rows):
        row    = (index // columns)
        col= index % columns
        img = protos[col][row].reshape(28,28)
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, index+1) )
        if index < 3:
            ax[-1].set_title("Restart #" + str(index))  # set title
        ax[-1].axis('off')
        if col == 0:
            ax[-1].text(-6, row+5, row)
        plt.imshow(img, cmap='Greys_r')
        index += 1
        # if c == 0:
        #     ax[-1].ylabel("Cluster #" + str(r))

plt.savefig("2-4-standard.png")
plt.show()  # finally, render the plot

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(large_dataset[0].reshape(28,28), cmap='Greys_r')
plt.show()


class HAC(object):
	def __init__(self, linkage):
		self.linkage = linkage
		
	def min_dist(self, c1, c2):
		# print("C1 shape", c1.shape)
		# print("C2 shape", c2.shape)
		return np.min(cdist(c1, c2))

	def max_dist(self, c1, c2):
		# print("Old c1", c1.shape)
		# print("C1 shape", c1.shape)
		# print("C2 shape", c2.shape)
		return np.max(cdist(c1, c2))

	def cent_dist(self, c1, c2):
		# print(c1.shape)
		# print(c2.shape)
		cent1 = np.mean(c1, axis = 0)
		cent2 = np.mean(c2, axis = 0)
		if cent1.ndim < 2:
			cent1 = np.expand_dims(cent1, axis=0)
		if cent2.ndim < 2:
			cent2 = np.expand_dims(cent2, axis=0)
		# print(cent1.shape)
		# print(cent2.shape)
		return np.linalg.norm(cent1 - cent2)

	def fit(self, X):
		# print(X.shape)
		# time.sleep(5)
		clusters = [np.expand_dims(image, axis=0) for image in X]
		numc = len(clusters)
		self.dists = []
		self.num_merges = []
		num_merges = 0
		while numc > 10:
			min_dist = float("Inf")
			best_c1 = None
			best_c2 = None
			# print(len(clusters))
			for i in range(numc-1):
				# print(i)
				for j in range(i+1, numc):
					if self.linkage == "min":
						dist = self.min_dist(clusters[i], clusters[j])
					elif self.linkage == "max":
						dist = self.max_dist(clusters[i], clusters[j])
					elif self.linkage == "centroid":
						dist = self.cent_dist(clusters[i], clusters[j])
					# print("Dist", dist)
					if dist == 0:
						print(clusters[i]==clusters[j])
					if dist < min_dist or best_c1 is None:
						# print("Smaller dist", dist)
						min_dist = dist
						# if dist == 0:
							# print(clusters[i] == clusters[j])
						best_c1 = i
						best_c2 = j
			# print("Min dist", min_dist)
			# print("Numc", numc)
			# print(best_c1)
			# print(best_c2)
			popped = clusters.pop(best_c2)
			if best_c1>best_c2:
				best_c1 -= 1
			clusters[best_c1] = np.vstack((clusters[best_c1], popped))
			num_merges += 1
			self.dists.append(min_dist)
			self.num_merges.append(num_merges)
			numc -= 1
			# print(len(clusters))
		self.clusters = clusters

	def get_mean_images(self):
		means = np.zeros((len(self.clusters), 784))
		for i in range(len(self.clusters)):
			# print(self.clusters[i].shape)
			means[i] = np.mean(self.clusters[i], axis=0)
		return means

	def get_cluster_sizes(self):
		sizes = []
		for i in range(len(self.clusters)):
			sizes.append(self.clusters[i].shape[0])
		return sizes

# Question 2.5
	
w = 10
h = 10
fig = plt.figure(figsize=(7, 9))
columns = 3
rows = 10

# ax enables access to manipulate each of subplots
ax = []
protos = np.zeros((3, 10, 784))
# test_dataset = small_dataset[np.random.choice(small_dataset.shape[0], 20, replace=False), :]
print(small_dataset.shape)
# print(test_dataset.shape)

HACmin = HAC("min")
HACmin.fit(small_dataset)
protos[0] = HACmin.get_mean_images()
print("min done")
HACmax = HAC("max")
HACmax.fit(small_dataset)
protos[1] = HACmax.get_mean_images()
print("max done")
HACcent = HAC("centroid")
HACcent.fit(small_dataset)
protos[2] = HACcent.get_mean_images()
# print(protos)

index = 0
for i in range(columns*rows):
		row = (index // columns)
		col = index % columns
		img = protos[col][row].reshape(28,28)
		# create subplot and append to ax
		ax.append(fig.add_subplot(rows, columns, index+1) )
		if index == 0:
			ax[-1].set_title("min")  # set title
		elif index == 1:
			ax[-1].set_title("max")  # set title
		elif index == 2:
			ax[-1].set_title("centroid")  # set title
		ax[-1].axis('off')
		if col == 0:
			ax[-1].text(-6, row+5, row)
		plt.imshow(img, cmap='Greys_r')
		index += 1
		# if c == 0:
		#     ax[-1].ylabel("Cluster #" + str(r))

plt.savefig("2-5-hacmeans.png")
plt.show()  # finally, render the plot



# Question 2.6
ypoints = HACmin.dists
xpoints = HACmin.num_merges
plt.plot(xpoints, ypoints)
plt.title("HAC Min Merge Dists")
plt.xlabel("Total number of merges completed")
plt.ylabel("Distance between most recently merged clusters")
plt.savefig("2-6-mindists.png")
plt.show()

ypoints = HACmax.dists
xpoints = HACmax.num_merges
plt.plot(xpoints, ypoints)
plt.title("HAC Max Merge Dists")
plt.xlabel("Total number of merges completed")
plt.ylabel("Distance between most recently merged clusters")
plt.savefig("2-6-maxdists.png")
plt.show()

ypoints = HACcent.dists
print(ypoints)
xpoints = HACcent.num_merges
plt.plot(xpoints, ypoints)
plt.title("HAC Centroid Merge Dists")
plt.xlabel("Total number of merges completed")
plt.ylabel("Distance between most recently merged clusters")
plt.savefig("2-6-centdists.png")
plt.show()

# Question 2.7
xpoints = list(range(10))

ypoints = HACmin.get_cluster_sizes()
plt.plot(xpoints, ypoints)
plt.title("HAC Min Images Per Cluster")
plt.xlabel("Cluster index”")
plt.ylabel("”Number of images in cluster")
plt.savefig("2-7-minims.png")
plt.show()

ypoints = HACmax.get_cluster_sizes()
plt.plot(xpoints, ypoints)
plt.title("HAC Max Images Per Cluster")
plt.xlabel("Cluster index”")
plt.ylabel("”Number of images in cluster")
plt.savefig("2-7-maxims.png")
plt.show()

# Question 2.8 - HAC

minclusters = HACmin.clusters
mat = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		# number of times that an image with  the  true  label  of j appears  in  cluster i
		ci = minclusters[i]
		labelsi = [small_labels[np.where((small_dataset == c).all(axis=1))[0]] for c in ci]
		mat[i][j] = labelsi.count(j)
heatmap(mat)
plt.title("HAC Min Heatmap")
plt.ylabel("Clusters")
plt.xlabel("True Label")
plt.savefig("2-8-hacminheat.png")
plt.show()

maxclusters = HACmax.clusters
mat = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		# number of times that an image with  the  true  label  of j appears  in  cluster i
		ci = maxclusters[i]
		labelsi = [small_labels[np.where((small_dataset == c).all(axis=1))[0]] for c in ci]
		mat[i][j] = labelsi.count(j)
heatmap(mat)
plt.title("HAC Max Heatmap")
plt.ylabel("Clusters")
plt.xlabel("True Label")
plt.savefig("2-8-hacmaxheat.png")
plt.show()

centclusters = HACcent.clusters
mat = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		# number of times that an image with  the  true  label  of j appears  in  cluster i
		ci = centclusters[i]
		labelsi = [small_labels[np.where((small_dataset == c).all(axis=1))[0]] for c in ci]
		mat[i][j] = labelsi.count(j)
heatmap(mat)
plt.title("HAC Centroid Heatmap")
plt.ylabel("Clusters")
plt.xlabel("True Label")
plt.savefig("2-8-haccentheat.png")
plt.show()

# Question - K Means

KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(small_dataset, get_cluster_ims=True)
kmeans_cluster_ims = KMeansClassifier.cluster_images
mat = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		# number of times that an image with  the  true  label  of j appears  in  cluster i
		ci = kmeans_cluster_ims[i]
		labelsi = [small_labels[np.where((small_dataset == c).all(axis=1))[0]] for c in ci]
		mat[i][j] = labelsi.count(j)
heatmap(mat)
plt.title("KMeans Heatmap")
plt.ylabel("Clusters")
plt.xlabel("True Label")
plt.savefig("2-8-kmeansheat.png")
plt.show()

# # This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
# plt.figure()
# plt.imshow(large_dataset[0].reshape(28,28), cmap='Greys_r')
# plt.show()
