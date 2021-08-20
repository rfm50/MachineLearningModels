# Predicting varieties in iris dataset using KNN classifier 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('iris.csv')

# Classifier will use sepal length and petal length as features
X = df[['sepal.length', 'petal.length']]
y = df['variety']

normal_X = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(normal_X, y, test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train) # Using 11 neighbors
y_pred = clf.predict(X_test)

f1_score = f1_score(y_test, y_pred, average='micro')
print(f'F1 score: {score:.2f}')
