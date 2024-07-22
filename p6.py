import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('./glass.csv')
df.head()
df.isna().sum()
X = df.drop('Type', axis=1)
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
clf = KNeighborsClassifier(n_neighbors=3, metric=euclidean_distance)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
clf = KNeighborsClassifier(n_neighbors=3, metric=manhattan_distance)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
