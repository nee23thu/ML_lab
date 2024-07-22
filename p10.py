import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA 
X = load_iris().data
y=load_iris().target
scaler = StandardScaler()
X = scaler.fit_transform(X)
sns.heatmap(np.corrcoef(X.T), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (After Standardization)')
plt.show()
pca = PCA(n_components=2)
x_proj = pca.fit_transform(X)
pc1 = x_proj[:,0]
pc2 = x_proj[:,1]
plt.scatter(pc1, pc2, c=y, cmap='jet')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
X = load_iris().data
y = load_iris().target
scaler = StandardScaler()
X = scaler.fit_transform(X)
sns.heatmap(np.corrcoef(X.T), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (After Standardization)')
plt.show()
lda = LinearDiscriminantAnalysis(n_components=2)
x_proj = lda.fit_transform(X,y)
ld1 = x_proj[:,0]
ld2 = x_proj[:,1]
plt.scatter(ld1, ld2,c=y, cmap='jet')
plt.show()
