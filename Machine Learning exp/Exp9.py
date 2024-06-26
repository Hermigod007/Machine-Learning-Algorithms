from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#generating dataset for regression
X=np.linspace(-5,5,100)
y=np.exp(X)

#plotting overall dataset
# plt.figure(figsize=(12, 8))
plt.xlabel("x")
plt.ylabel("f(x)=e^x")
plt.title("Overall Dataset")
plt.plot(X,y)

#splitting the dataset 80-20 rule
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# plotting splitted dataset
plt.figure(figsize=(8,5))
plt.xlabel("x")
plt.ylabel("f(x)=e^x")
plt.title("Splitted Dataset")
plt.scatter(X_train, y_train, marker='^', label="training data set")
plt.scatter(X_test, y_test, marker='*', label="testing data set")
plt.legend()

DecisionTree=DecisionTreeRegressor()
print(X_train.shape)
DecisionTree.fit(X_train.reshape(-1,1),y_train)

#trained tree plotting
plt.figure(figsize=(20, 20))
plot_tree(DecisionTree,filled=True)
plt.title("Trained Decision Tree")
plt.show()

y_pred=DecisionTree.predict(X_test.reshape(-1,1))
print(y_pred)
print(y_test)
result=mean_squared_error(y_true=y_test,y_pred=y_pred)
print("\nRoot Mean Squared Error on testing data set: ", result)

y_pred_train=DecisionTree.predict(X_train.reshape(-1,1))
result=mean_squared_error(y_true=y_train,y_pred=y_pred_train)
print("\nRoot Mean Squared Error on training data set: ", result)



