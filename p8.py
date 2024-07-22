import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
X = load_iris().data
y = load_iris().target
# or
# df = pd.read_csv('./iris.csv')
# X = np.array(df.iloc[:,1:-1].values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
def kmeans(X, K, max_iters):
    centroids = X[:K]
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis]-centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(K)])
        if np.all(new_centroids==centroids):
            break
        centroids=new_centroids
    return labels, centroids
labels, c = kmeans(X, 3, 20)
plt.scatter(X[:, 0], X[:,1],c=labels)
plt.scatter(c[:,0], c[:,1], marker='x', color='red')
plt.show()
sns.heatmap(confusion_matrix(y, labels), annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
