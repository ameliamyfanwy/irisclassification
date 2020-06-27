import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

# Split data set into features
# features = iris.data.T

# Assign features to variables
# sepal_length = features[0]
# sepal_width = features[1]
# petal_length = features[2]
# petal_width = features[3]

# Assign names to variables for use when plotting
# sepal_length_label = iris.feature_names[0]
# sepal_width_label = iris.feature_names[1]
# petal_length_label = iris.feature_names[2]
# petal_width_label = iris.feature_names[3]

# Create plot
# plt.scatter(petal_length, petal_width, c=iris.target)
# plt.xlabel(petal_length_label)
# plt.ylabel(petal_width_label)
# plt.show()

# Splitting data for training using train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# Set parameter for how many neighbours to compare to
knn = KNeighborsClassifier(n_neighbors=1)

# Use training data to produce best fit function
knn.fit(X_train, y_train)