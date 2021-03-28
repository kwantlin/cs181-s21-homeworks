import numpy as np

clusters = np.array([[2,3], [3,4], [5,6]])
print(clusters)
clusters = np.delete(clusters, np.argwhere( clusters == [5,6], axis = 0), axis=0)
print(clusters)