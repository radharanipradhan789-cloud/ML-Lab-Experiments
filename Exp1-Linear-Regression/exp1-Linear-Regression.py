import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Data
# -------------------------
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

# -------------------------
# Linear Regression
# -------------------------
lin_model = LinearRegression()
lin_model.fit(X, Y)
Y_lin_pred = lin_model.predict(X)

# Linear metrics
mae_lin = mean_absolute_error(Y, Y_lin_pred)
mse_lin = mean_squared_error(Y, Y_lin_pred)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(Y, Y_lin_pred)
n, p = len(Y), 1
adj_r2_lin = 1 - (1 - r2_lin) * (n - 1) / (n - p - 1)

# -------------------------
# Polynomial Regression (degree 2)
# -------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)
Y_poly_pred = poly_model.predict(X_poly)

# Polynomial metrics
mae_poly = mean_absolute_error(Y, Y_poly_pred)
mse_poly = mean_squared_error(Y, Y_poly_pred)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(Y, Y_poly_pred)
adj_r2_poly = 1 - (1 - r2_poly) * (n - 1) / (n - p - 1)

# -------------------------
# Print Metrics
# -------------------------
print("--- Linear Regression ---")
print(f"MAE: {mae_lin:.2f}, MSE: {mse_lin:.2f}, RMSE: {rmse_lin:.2f}, R2: {r2_lin:.2f}, Adjusted R2: {adj_r2_lin:.2f}")
print("--- Polynomial Regression ---")
print(f"MAE: {mae_poly:.2f}, MSE: {mse_poly:.2f}, RMSE: {rmse_poly:.2f}, R2: {r2_poly:.2f}, Adjusted R2: {adj_r2_poly:.2f}")

print("\n--- Error Comparison ---")
print("Metric\t\tLinear\tPolynomial")
print(f"MAE\t\t{mae_lin:.2f}\t{mae_poly:.2f}")
print(f"MSE\t\t{mse_lin:.2f}\t{mse_poly:.2f}")
print(f"RMSE\t\t{rmse_lin:.2f}\t{rmse_poly:.2f}")
print(f"R2\t\t{r2_lin:.2f}\t{r2_poly:.2f}")

# -------------------------
# Visualization (4 plots in one figure)
# -------------------------
plt.figure(figsize=(14, 10))

# 1. Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_lin_pred, color='red', label='Linear Regression')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()

# 2. Linear vs Polynomial Regression
plt.subplot(2, 2, 2)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_lin_pred, color='red', label='Linear Regression')
plt.plot(X, Y_poly_pred, color='green', label='Polynomial Regression')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()

# 3. Predicted vs Actual
plt.subplot(2, 2, 3)
plt.scatter(Y, Y_lin_pred, color='red', label='Linear Predicted')
plt.scatter(Y, Y_poly_pred, color='green', label='Polynomial Predicted')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='blue', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Predicted vs Actual")
plt.legend()

# 4. Error Metrics Comparison (Bar plot)
plt.subplot(2, 2, 4)
metrics = ['MAE', 'MSE', 'RMSE']
linear_errors = [mae_lin, mse_lin, rmse_lin]
poly_errors = [mae_poly, mse_poly, rmse_poly]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, linear_errors, width, label='Linear')
plt.bar(x + width/2, poly_errors, width, label='Polynomial')
plt.xticks(x, metrics)
plt.ylabel("Error Value")
plt.title("Error Metrics Comparison")
plt.legend()

plt.tight_layout()
plt.show()
