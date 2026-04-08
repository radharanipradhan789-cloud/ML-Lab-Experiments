import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Given data
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate coefficients
b1 = np.sum((x-x_mean)*(y-y_mean)) / np.sum((x-x_mean)**2)
b0 = y_mean - b1*x_mean

# Predictions
y_pred = b0 + b1*x

# Metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
n = len(x)
p = 1
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

print("Regression Line: ŷ =", round(b0,3), "+", round(b1,3),"x")
print("MSE:", round(mse,3))
print("MAE:", round(mae,3))
print("RMSE:", round(rmse,3))
print("R²:", round(r2,3))
print("Adjusted R²:", round(adj_r2,3))

# Visualization
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, color='red', label="Best Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Best Fit Line")
plt.legend()
plt.show()

#Output
Regression Line: ŷ = 2.2 + 0.6 x
MSE: 0.48
MAE: 0.64
RMSE: 0.693
R²: 0.6
Adjusted R²: 0.467
