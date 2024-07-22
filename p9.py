import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
data=load_iris().data[:6]
def proximity_matrix(data):
    n = data.shape[0]
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            mat[i,j] = np.linalg.norm(data[i]-data[j])
            mat[j,i]=mat[i,j]
    return mat
proximity_matrix(X)
def plot_dendrogram(data, method):
    mat = linkage(data, method=method)
    dendrogram(mat)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()
plot_dendrogram(data,'single')
plot_dendrogram(data,'complete')
