import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINE DATASET 
# ==========================================
# Design matrix X
X = np.array([
    [1, 2],
    [1, 3],
    [1, 4]
])

# Target variables
y1 = np.array([1, 2, 3])
y2 = np.array([2, 4, 6])

# Regularization Parameter
lambda_val = 1

# Ridge Regression formula: Beta = (X^T @ X + lambda * I)^-1 @ X^T @ Y
I = np.eye(X.shape[1])

# Calculating Ridge coefficients (beta)
beta_y1 = np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y1
beta_y2 = np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y2

print("Ridge coefficients for y1:", beta_y1)
print("Ridge coefficients for y2:", beta_y2)

# Predictions
y1_pred = X @ beta_y1
y2_pred = X @ beta_y2

# ==========================================
# 2. VISUALIZATION 
# ==========================================

# Y1 Actual vs Predicted
plt.figure()
plt.plot(y1, marker='o', label="Actual y1")
plt.plot(y1_pred, marker='x', label="Predicted y1")
plt.xlabel("Observation")
plt.ylabel("y1")
plt.legend()
plt.show()

# Y2 Actual vs Predicted
plt.figure()
plt.plot(y2, marker='o', label="Actual y2")
plt.plot(y2_pred, marker='x', label="Predicted y2")
plt.xlabel("Observation")
plt.ylabel("y2")
plt.legend()
plt.show()

# Y1 vs Y2
plt.figure()
plt.plot(y1, y2, marker='o')
plt.title("Graph 3: y1 vs y2")
plt.xlabel("y1")
plt.ylabel("y2")
plt.show()

# ==========================================
# 3. PREDICTED VS ACTUAL & IDEAL LINE 
# ==========================================

# Predicted vs Actual (y1)
plt.figure()
plt.scatter(y1, y1_pred, label="Predicted Points")

# Ideal line (Perfect Prediction line: y = x)
plt.plot(y1, y1, linestyle='--', label="Ideal line (y=x)", color='red')
plt.title("Predicted vs Actual (y1)")
plt.xlabel("Actual y1")
plt.ylabel("Predicted y1")
plt.legend()
plt.show()

#Output
Ridge coefficients for y1: [-4.44089210e-16  6.66666667e-01]
Ridge coefficients for y2: [-8.88178420e-16  1.33333333e+00]
