import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Data Points (x1, x2)
X = np.array([[1, 1], [2, 1], [2, 3], [3, 3]])

# Target Labels (y)
# Note: SKlearn expects labels to be -1 and 1, which matches the image.
y = np.array([1, 1, -1, -1])

# New point to classify
new_point = np.array([[2, 2]])

# Create a linear SVM Classifier
# Using a high C value (1000) for a "hard margin" approach
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Get weights (w) and bias (b)
w = clf.coef_[0]
b = clf.intercept_[0]

print(f"Optimal weights (w): {w}")
print(f"Optimal bias (b): {b}")

# Display the optimal equation: w0*x1 + w1*x2 + b = 0
print(f"Optimal equation: {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {b:.2f} = 0")

# Calculate the Margin: 2 / ||w||
margin = 2 / np.sqrt(np.sum(w**2))
print(f"Margin: {margin:.4f}")

# Prediction for the new point
prediction = clf.predict(new_point)
print(f"Prediction for {new_point}: {prediction}")

#Output
Optimal weights (w): [-6.4000e-04 -9.9968e-01]
Optimal bias (b): 2.0007466666666667
Optimal equation: -0.00*x1 + -1.00*x2 + 2.00 = 0
Margin: 2.0006
Prediction for [[2 2]]: [1]
