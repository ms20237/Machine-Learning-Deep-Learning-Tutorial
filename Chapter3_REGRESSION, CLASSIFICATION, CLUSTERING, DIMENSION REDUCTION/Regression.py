import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics


# Generate random points
np.random.seed(0)  # For reproducibility
x = 2 * np.random.rand(200, 1)
y = 4 + 3 * x + np.random.randn(200, 1)

# Plot points
plt.scatter(x, y)
plt.show()


# Adding x0 = 1 to each instance for the intercept term
x_b = np.c_[np.ones((200, 1)), x]  # Add bias term (x0 = 1)

# Normal Equation
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print("Coefficients (theta):", theta_best)


# Making predictions
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]
y_predict = x_new_b.dot(theta_best)

# Visualize the fit
plt.plot(x_new, y_predict, "r-", label="Predictions")
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit (from scratch)")
plt.legend()
plt.show()



# use Scikit-learn package
X = np.array(x).reshape(-1, 1)  # Convert to a 2D array
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a linear regression model and fit it to the training data:
model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# Make predictions on the test data:
y_pred = model.predict(X_test)

# Evaluate the model's performance using a metric like mean squared error (MSE):
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# Plot the data points and the regression line:
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()
