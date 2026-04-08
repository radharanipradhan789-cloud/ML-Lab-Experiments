import numpy as np

# --- Manual SVM Weight Calculation ---

# Lagrange multipliers (alpha)
alpha = np.array([1, 1, 1])

# Support vectors (the points that define the margin)
support_vectors = np.array([
    [0, -1, -1],
    [0, 2, -1],
    [-1, 0, 2]
])

# Labels for the support vectors
y = np.array([1, -1, -1])

# New test point
x = np.array([0.2, 0.8, 0.4])

# Calculate weight vector w = sum(alpha_i * y_i * support_vector_i)
# Correction: Using np.newaxis to allow proper broadcasting with the arrays
w = np.sum((alpha[:, np.newaxis] * y[:, np.newaxis]) * support_vectors, axis=0)

# Decision function: f(x) = w dot x
f_x = np.dot(w, x)

# Predicted class: sign of f(x)
y_pred = np.sign(f_x)
print(f"Weight vector (w): {w}")
print(f"Decision value f(x): {f_x:.4f}")
print(f"Predicted Class: {int(y_pred)}")

#Output
Weight vector (w): [ 1 -3 -2]
Decision value f(x): -3.0000
Predicted Class: -1
