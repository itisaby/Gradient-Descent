from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading Datasets
iris = datasets.load_iris()

# print(iris.DESCR)

features = iris.data
labels = iris.target
print(features[0], labels[0])

#Training the data
clf = KNeighborsClassifier()

clf.fit(features, labels)

#Prediction
pred = clf.predict([[9.1, 3.5, 6.4, 4.8]])
print(pred)
