import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for visualization
from sklearn.datasets import fetch_openml  # Importing function to fetch MNIST dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Importing Linear Discriminant Analysis (LDA)

# Load MNIST dataset
mnist = fetch_openml('mnist_784')  # Fetching the MNIST dataset
X, y = mnist['data'].values, mnist['target'].values  # Extracting data and labels

# Convert string labels to integers
y = y.astype(np.uint8)  # Converting labels to unsigned integers

# Select images of digits 4, 7, and 8
selected_digits = [4, 7, 8]  # Defining the selected digits
selected_indices = np.where(np.isin(y, selected_digits))[0]  # Finding indices of selected digits
X_selected = X[selected_indices]  # Selecting images corresponding to selected digits
y_selected = y[selected_indices]  # Selecting labels corresponding to selected digits

# Shuffle data
np.random.seed(42)  # Setting random seed for reproducibility
shuffle_indices = np.random.permutation(len(X_selected))  # Generating shuffled indices
X_selected_shuffled = X_selected[shuffle_indices]  # Shuffling images
y_selected_shuffled = y_selected[shuffle_indices]  # Shuffling labels

# Take the first 900 images for training
X_train = X_selected_shuffled[:900]  # Selecting training images
y_train = y_selected_shuffled[:900]  # Selecting training labels

# Initialize Linear Discriminant Analysis (LDA) object
lda = LinearDiscriminantAnalysis()  # Creating LDA object

# Fit LDA model
lda.fit(X_train, y_train)  # Fitting LDA model to the training data

# Get explained variance ratios for all LDA dimensions
explained_variance_ratios = lda.explained_variance_ratio_  # Extracting explained variance ratios

# Plot explained variance ratios
plt.figure(figsize=(10, 6))  # Setting figure size
plt.plot(np.cumsum(explained_variance_ratios), marker='o')  # Plotting cumulative explained variance ratios
plt.title('Explained Variance Ratios for LDA Dimensions')  # Setting plot title
plt.xlabel('Number of LDA Dimensions')  # Setting x-axis label
plt.ylabel('Cumulative Explained Variance Ratio')  # Setting y-axis label
plt.grid(True)  # Adding grid lines
plt.show()  # Displaying the plot

# Maximum LDA dimensions with non-zero explained variance ratios
max_lda_dimensions = np.count_nonzero(explained_variance_ratios)  # Counting non-zero explained variance ratios
print(f"Maximum LDA dimensions with non-zero explained variance ratios: {max_lda_dimensions}")  # Printing the result

# Reasoning: LDA can have at most C - 1 dimensions, where C is the number of classes (in this case, C = 3).
