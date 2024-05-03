
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x * np.exp(-x**2)

def gradient(x):
    return (1 - 2*x**2) * np.exp(-x**2)

# Given range and parameters
x_range = np.linspace(-np.sqrt(1.5), 0, 100)
initial_x = -np.sqrt(1.5)
threshold = 0.001
step_sizes = [0.001, 0.005, 0.01, 0.05]

def gradient_descent(initial_x, step_size, threshold):
    x = initial_x
    iterations = 0
    while np.linalg.norm(gradient(x)) > threshold:
        x = x - step_size * gradient(x)
        iterations += 1

    minima_value = objective_function(x)
    return x, minima_value, iterations

# Plotting the function
plt.plot(x_range, objective_function(x_range), label='f(x) = x * e^(-x^2)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Plot')
plt.legend()
plt.show()

# Table header
print("{:<10} {:<15} {:<20} {:<25}".format('Step Size', 'Iterations', 'Minima Value', 'Function Value at Minima'))

# Gradient Descent for different step sizes
for step_size in step_sizes:
    result = gradient_descent(initial_x, step_size, threshold)
    print("{:<10} {:<15} {:<20} {:<25}".format(step_size, result[2], result[0], result[1]))

# Analytical solution
analytical_solution = -1/np.sqrt(2)
analytical_value = objective_function(analytical_solution)
print("\nAnalytical Solution:")
print("Minima Value: {:<20}".format(analytical_solution))
print("Function Value at Minima: {:<25}".format(analytical_value))

# Define the bivariate function
def f(x1, x2):
    return 10*x1**2 + 5*x1*x2 + 10*(x2-3)**2

# Define the gradient of the function
def gradient_f(x):
    return np.array([20*x[0] + 5*x[1], 5*x[0] + 20*(x[1]-3)])

# Define the gradient descent algorithm
def gradient_descent(x0, eta, threshold):
    x = x0
    iterations = 0
    while True:
        grad = gradient_f(x)
        if np.linalg.norm(grad) < threshold:
            break
        x = x - eta * grad
        iterations += 1
    return x, f(x[0], x[1]), iterations

x0 = np.array([10, 15])

# Step size values
etas = [0.001, 0.005, 0.01, 0.05]

# Store results in a list of tuples (step_size, iterations, minima_value, function_value_at_minima)
results = []

# Iterate over different step sizes
for eta in etas:
    minima, minima_value, iterations = gradient_descent(x0, eta, 0.001)
    results.append((eta, iterations, minima_value, f(minima[0], minima[1])))

# Print the results as a table
print("Step Size | Iterations | Minima Value | Function Value at Minima")
print("-" * 50)
for result in results:
    print("{:<9} | {:<11} | {:<13} | {:<25}".format(*result))

# Plot the function
x1_range = np.linspace(-10, 10, 100)
x2_range = np.linspace(-15, 15, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Surface Plot of the Function')
plt.show()