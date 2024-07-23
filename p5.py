import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./TitanicDataset.csv')
df.head()
df = df[['Survived','Pclass','Age','SibSp','Parch','Fare','Embarked']]
df.isna().sum()
df['Age']=SimpleImputer(strategy='median').fit_transform(df[['Age']])
df['Embarked']=LabelEncoder().fit_transform(df[['Embarked']])
df.isna().sum()
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
