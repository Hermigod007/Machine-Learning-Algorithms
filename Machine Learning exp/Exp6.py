import numpy as np  # Import NumPy library for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.datasets import fetch_openml  # Import function to load dataset
from sklearn.decomposition import PCA  # Import PCA (Principal Component Analysis) from scikit-learn

# Load MNIST dataset
mnist = fetch_openml('mnist_784')  # Load MNIST dataset with 784 features
X, y = mnist['data'].values, mnist['target'].values  # Separate features and labels

# Convert string labels to integers
y = y.astype(np.uint8)

# Select images of digits 4, 7, and 8
selected_digits = [4, 7, 8]
selected_indices = np.where(np.isin(y, selected_digits))[0]  # Find indices of selected digits
X_selected = X[selected_indices]  # Select images corresponding to selected digits

# Shuffle data
np.random.shuffle(X_selected)

# Take the first 900 images for training
X_train = X_selected[:900]

# Compute mean image
mean_image = np.mean(X_train, axis=0)

# Compute sample covariance matrix
cov_matrix = np.cov(X_train, rowvar=False)

# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Plot eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues, marker='o')  # Plot eigenvalues
plt.title('Eigenvalues of Sample Covariance Matrix')  # Set title
plt.xlabel('Eigenvalue Index')  # Set x-axis label
plt.ylabel('Eigenvalue')  # Set y-axis label
plt.grid(True)  # Add grid
plt.show()  # Display plot

# Visualize original and reconstructed images for different PCA dimensions
pca_dimensions = [2, 10, 50, 100, 200, 300]
for m in pca_dimensions:
    # PCA projection matrix
    A = eigenvectors[:, :m]  # Select top m eigenvectors as projection matrix

    # Project images onto the PCA space
    X_projected = np.dot(X_train - mean_image, A)  # Project images onto PCA space
    X_reconstructed = np.dot(X_projected, A.T) + mean_image  # Reconstruct images from projected data

    # Visualize original and reconstructed images
    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axes[0, i].imshow(X_train[i].reshape(28, 28), cmap='gray')  # Display original images
        axes[0, i].axis('off')  # Turn off axis
        axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')  # Display reconstructed images
        axes[1, i].axis('off')  # Turn off axis
    fig.suptitle(f'PCA Dimension: {m}')  # Set title
    plt.show()  # Display plot

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# Determine minimum dimensions to keep 98% of total variance
min_dimensions = np.argmax(cumulative_variance_ratio >= 0.98) + 1
print(f"At least {min_dimensions} dimensions are required to keep 98% of the total variance.")
