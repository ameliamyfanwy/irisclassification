import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris

iris = load_iris()

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

plt.scatter(petal_length, petal_width, c=iris.target)
plt.xlabel(petal_length_label)
plt.ylabel(petal_width_label)
plt.show()