import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

x, y = make_regression(n_samples=100, n_features=1, noise=5, bias=0)

# Scale the data
x_scaled = np.interp(x, (x.min(), x.max()), (-5, 5))
y_scaled = np.interp(y, (y.min(), y.max()), (-15, 15))

# Add bias term to input features
x_with_bias = np.c_[np.ones(x_scaled.shape[0]), x_scaled]
train_x, test_x, train_y, test_y = train_test_split(x_with_bias, y_scaled, train_size=0.8, test_size=0.2)

# Visualize the data
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

# List of alpha values for Ridge regularization
alphas = [0.1, 0.5, 1, 1.5]

for alpha in alphas:
    model = Ridge(alpha)
    model.fit(train_x[:, 1:], train_y)
    # Obtain model parameters
    intercept = model.intercept_
    slope = model.coef_[0]

# Make predictions on training and testing data
y_train_pred_ridge = model.predict(train_x[:, 1:])
y_test_pred_ridge = model.predict(test_x[:, 1:])

# Calculate RMSE for training and testing data
rmse_train_ridge = np.sqrt(mean_squared_error(train_y, y_train_pred_ridge))
rmse_test_ridge = np.sqrt(mean_squared_error(test_y, y_test_pred_ridge))

# Print results for the current alpha value
print(f"\nRidge Regression with alpha = {alpha}:")
print("Root Mean Squared Error (RMSE) for training data:", rmse_train_ridge)
print("Root Mean Squared Error (RMSE) for testing data:", rmse_test_ridge)
print("Coefficients:")
print("Intercept:", intercept)
print("Slope:", slope)

# Verification of coefficients using the Ridge model's attributes
print("\nVerification with Ridge function:")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])