from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Selecting petal length and width features only
y = iris.target

# Display dataset
print("Dataset:")
print("Petal Length | Petal Width | Species")
for i in range(len(X)):
    print(f"{X[i][0]:.2f}          {X[i][1]:.2f}          {iris.target_names[y[i]]}")

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train logistic regression multiclass algorithm using Softmax Regression
log_reg = LogisticRegression(multi_class='multinomial')
log_reg.fit(X_train, y_train)

# Test the logistic regression algorithm on testing dataset
y_pred_log_reg = log_reg.predict(X_test)

# Display results for logistic regression
print("\nResults for Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=iris.target_names))

# Train linear SVM algorithm
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Test the linear SVM algorithm on testing dataset
y_pred_linear_svm = linear_svm.predict(X_test)

# Display results for linear SVM
print("\nResults for Linear SVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_linear_svm))# Import necessary libraries for data loading, model training, and evaluation
from sklearn.datasets import load_iris  # Import function to load the Iris dataset
from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score  # Import function to calculate accuracy

# Load Iris dataset
iris = load_iris()  # Load the Iris dataset into the variable 'iris'

# Display dataset
print("IRIS Dataset:")  # Print statement for dataset visualization
print(iris.data)  # Print the features of the Iris dataset

# Use petal length and width features only
X = iris.data[:, 2:]  # Selecting only the petal length and width features for input 'X'
y = iris.target  # Assigning target labels to 'y'

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets

# Train Logistic Regression multiclass algorithm
logreg_model = LogisticRegression(max_iter=1000, multi_class='multinomial')  # Initialize a Logistic Regression classifier
logreg_model.fit(X_train, y_train)  # Train the Logistic Regression model

# Test Logistic Regression model
logreg_pred = logreg_model.predict(X_test)  # Predict the target labels using the trained Logistic Regression model
logreg_accuracy = accuracy_score(y_test, logreg_pred)  # Calculate the accuracy of the Logistic Regression model

# Display results of training and testing of Logistic Regression algorithm
print("\nLogistic Regression Results:")  # Print statement for Logistic Regression results
print("Training Accuracy:", logreg_model.score(X_train, y_train))  # Print the training accuracy of the Logistic Regression model
print("Testing Accuracy:", logreg_accuracy)  # Print the testing accuracy of the Logistic Regression model

# Train linear SVM algorithm
svm_linear_model = SVC(kernel='linear')  # Initialize a linear Support Vector Classifier
svm_linear_model.fit(X_train, y_train)  # Train the linear SVM model

# Test linear SVM model
svm_linear_pred = svm_linear_model.predict(X_test)  # Predict the target labels using the trained linear SVM model
svm_linear_accuracy = accuracy_score(y_test, svm_linear_pred)  # Calculate the accuracy of the linear SVM model

# Display results of training and testing of linear SVM algorithm
print("\nLinear SVM Results:")  # Print statement for linear SVM results
print("Training Accuracy:", svm_linear_model.score(X_train, y_train))  # Print the training accuracy of the linear SVM model
print("Testing Accuracy:", svm_linear_accuracy)  # Print the testing accuracy of the linear SVM model

# Train soft SVM algorithm
svm_soft_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Initialize a soft SVM classifier with RBF kernel
svm_soft_model.fit(X_train, y_train)  # Train the soft SVM model

# Test soft SVM model
svm_soft_pred = svm_soft_model.predict(X_test)  # Predict the target labels using the trained soft SVM model
svm_soft_accuracy = accuracy_score(y_test, svm_soft_pred)  # Calculate the accuracy of the soft SVM model

# Display results of training and testing of soft SVM algorithm
print("\nSoft SVM Results:")  # Print statement for soft SVM results
print("Training Accuracy:", svm_soft_model.score(X_train, y_train))  # Print the training accuracy of the soft SVM model
print("Testing Accuracy:", svm_soft_accuracy)  # Print the testing accuracy of the soft SVM model

print("\nClassification Report:")
print(classification_report(y_test, y_pred_linear_svm, target_names=iris.target_names))

# Train soft SVM algorithm
soft_svm = SVC(kernel='rbf')
soft_svm.fit(X_train, y_train)

# Test the soft SVM algorithm on testing dataset
y_pred_soft_svm = soft_svm.predict(X_test)

# Display results for soft SVM
print("\nResults for Soft SVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_soft_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_soft_svm, target_names=iris.target_names))
