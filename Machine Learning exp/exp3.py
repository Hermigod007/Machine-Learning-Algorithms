import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron


x, Y = make_classification(n_samples=100, n_features=5, n_classes=2)

converted_Y = 2 * Y - 1
train_x, test_x, train_labels, test_labels = train_test_split(x, converted_Y, train_size=0.8, test_size=0.2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_x[:, 0], train_x[:, 1], c="green", label="Training Data")
plt.scatter(test_x[:, 0], test_x[:, 1], c="red", label="Testing Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Initial Scatter Plot for First Two Features")
plt.legend()


def perceptron_train(X_train, y_train, learning_rate=1):
    num_features = X_train.shape[1]  # Number of features
    weights = np.zeros(num_features)  # Initialize weights to zeros
    mistakes = 0

    # Iterating through each training sample
    for i in range(X_train.shape[0]):
        prediction = np.sign(np.dot(X_train[i], weights))  # Compute prediction
        if prediction != y_train[i]:
            mistakes += 1  # Increment mistake count if prediction is incorrect
            weights += learning_rate * y_train[i] * X_train[i]  # Update weights
        return weights, mistakes

# Train the perceptron and obtain final weights
final_weights, num_mistakes = perceptron_train(train_x, train_labels)
print("\nfinal weight vector:-")
print(final_weights)

# Predict class labels for the test set
y_pred = np.sign(np.dot(test_x, final_weights))

# Plot predicted data with decision boundary
plt.subplot(1, 2, 2)
plt.scatter(test_x[:, 0], test_x[:, 1], c="red", label="Testing Data")
plt.scatter(test_x[:, 0], test_x[:, 1], c=y_pred, marker="*", label="Predicted Data")
x_values = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
y_values = -(final_weights[0] * x_values) / final_weights[1]  # Compute decision boundary
plt.plot(x_values, y_values, label="Decision Boundary", color="blue")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Predicted Data with Decision Boundary")
plt.legend()
plt.tight_layout()
plt.show()

# Compute confusion matrix and related metrics
conf_matrix = confusion_matrix(test_labels, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
print(f"\nTrue Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")

# Compute misclassification error and accuracy
misclassification_error = (FP + FN) / len(test_labels)
accuracy = accuracy_score(test_labels, y_pred)
print(f"\nMisclassification Error: {misclassification_error:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
