import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

x, y = make_regression(n_samples=100, n_features=1, noise=5, bias=0)

x_scaled = np.interp(x, (x.min(), x.max()), (-5, 5))
y_scaled = np.interp(y, (y.min(), y.max()), (-15, 15))

# Adding bias term to features
x_with_bias = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

# Splitting data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x_with_bias, y_scaled, train_size=0.8, test_size=0.2)

# Visualizing the data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_scaled, y_scaled, c='blue', label='Overall Dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Overall Dataset')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(train_x[:, 1], train_y, c='green', label='Training Data')
plt.scatter(test_x[:, 1], test_y, c='red', label='Testing Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Train-Test Dataset')
plt.legend()
plt.tight_layout()
plt.show()

# Calculating coefficients using the normal equation
X_transpose = np.transpose(train_x)
coefficients = np.linalg.inv(X_transpose.dot(train_x)).dot(X_transpose).dot(train_y)
bias = coefficients[0]
slope = coefficients[1]

# Visualizing regression line
plt.figure(figsize=(10, 5))
plt.scatter(train_x[:, 1], train_y, c='green', label='Training Data')
plt.scatter(test_x[:, 1], test_y, c='red', label='Testing Data')
x_values = np.linspace(np.min(x_scaled), np.max(x_scaled), 100)
y_values = bias + slope * x_values
plt.plot(x_values, y_values, color='blue', label=f'y ={slope:.2f}x + {bias:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training and Testing Datasets with Regression Line')
plt.legend()
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(train_x[:, 1:], train_y)

# Obtaining model parameters
model_intercept = model.intercept_
model_slope = model.coef_[0]

# Making predictions
y_train_pred = model.predict(train_x[:, 1:])
y_test_pred = model.predict(test_x[:, 1:])

# Calculating RMSE for training and testing
rmse_train = np.sqrt(mean_squared_error(train_y, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(test_y, y_test_pred))

# Printing results
print("Root Mean Squared Error (RMSE) for training data:", rmse_train)
print("Root Mean Squared Error (RMSE) for testing data:", rmse_test)
print("Coefficients obtained from Linear Regression function:")
print(f"Intercept: {model_intercept}")
print(f"Slope: {model_slope}")
print("Verification:")
print("Coefficients obtained from model training:")
print(f"Intercept: {bias}")
print(f"Slope: {slope}")