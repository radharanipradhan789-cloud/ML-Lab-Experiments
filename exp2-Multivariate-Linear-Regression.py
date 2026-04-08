import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- Given data -----
x1 = np.array([1, 2, 3]).reshape(-1, 1)
Y = np.array([[2, 3],
              [4, 5],
              [6, 7]])  # Columns: Y1 and Y2

# ----- Add intercept for matrix calculation -----
X = np.hstack((np.ones((x1.shape[0], 1)), x1))

# ----- Multivariate Linear Regression using Matrix Approach -----
B = np.linalg.inv(X.T @ X) @ X.T @ Y
Y_pred = X @ B  # Predicted values

# ----- Function to calculate metrics -----
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true)-2)
    return mse, mae, rmse, r2, adj_r2

# ----- Calculate metrics for Y1 and Y2 -----
mse_Y1, mae_Y1, rmse_Y1, r2_Y1, adj_r2_Y1 = calculate_metrics(Y[:,0], Y_pred[:,0])
mse_Y2, mae_Y2, rmse_Y2, r2_Y2, adj_r2_Y2 = calculate_metrics(Y[:,1], Y_pred[:,1])

# ----- Print regression coefficients -----
print("Regression Coefficients (Intercept and Slope for Y1 and Y2):")
print(B)

# ----- Print error metrics -----
print("\nError Metrics for Y1:")
print(f"MSE: {mse_Y1}, MAE: {mae_Y1}, RMSE: {rmse_Y1}, R2: {r2_Y1}, Adj R2: {adj_r2_Y1}")

print("\nError Metrics for Y2:")
print(f"MSE: {mse_Y2}, MAE: {mae_Y2}, RMSE: {rmse_Y2}, R2: {r2_Y2}, Adj R2: {adj_r2_Y2}")

# ----- Print Predicted vs Actual Table -----
print("\nPredicted vs Actual Data:")
for i in range(len(x1)):
    print(f"x1={x1[i,0]} -> Actual Y1={Y[i,0]}, Predicted Y1={Y_pred[i,0]:.3f} | Actual Y2={Y[i,1]}, Predicted Y2={Y_pred[i,1]:.3f}")

# ----- Visualization -----

# 1. Actual vs Predicted Y1
plt.figure()
plt.scatter(x1, Y[:,0], color='blue', label='Actual Y1')
plt.plot(x1, Y_pred[:,0], color='red', label='Predicted Y1', linewidth=2)
plt.xlabel('X1')
plt.ylabel('Y1')
plt.title('Multivariate Regression: Y1')
plt.legend()
plt.grid(True)

# 2. Actual vs Predicted Y2
plt.figure()
plt.scatter(x1, Y[:,1], color='green', label='Actual Y2')
plt.plot(x1, Y_pred[:,1], color='orange', label='Predicted Y2', linewidth=2)
plt.xlabel('X1')
plt.ylabel('Y2')
plt.title('Multivariate Regression: Y2')
plt.legend()
plt.grid(True)

# 3. Error Metrics Comparison
metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'Adj R2']
Y1_metrics = [mse_Y1, mae_Y1, rmse_Y1, r2_Y1, adj_r2_Y1]
Y2_metrics = [mse_Y2, mae_Y2, rmse_Y2, r2_Y2, adj_r2_Y2]
x_pos = np.arange(len(metrics))

plt.figure()
plt.bar(x_pos - 0.15, Y1_metrics, width=0.3, color='red', label='Y1')
plt.bar(x_pos + 0.15, Y2_metrics, width=0.3, color='green', label='Y2')
plt.xticks(x_pos, metrics)
plt.ylabel('Metric Value')
plt.title('Error Metrics Comparison (Y1 vs Y2)')
plt.legend()
plt.grid(True, axis='y')

# 4. Predicted vs Actual Scatter
plt.figure()
plt.scatter(Y[:,0], Y_pred[:,0], color='red', label='Y1 Predicted')
plt.scatter(Y[:,1], Y_pred[:,1], color='green', label='Y2 Predicted')
plt.plot([min(Y.flatten()), max(Y.flatten())],
         [min(Y.flatten()), max(Y.flatten())],
         color='blue', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Predicted vs Actual (Multivariate)')
plt.legend()
plt.grid(True)
plt.show()

#Output
Regression Coefficients (Intercept and Slope for Y1 and Y2):
[[-3.55271368e-15  1.00000000e+00]
 [ 2.00000000e+00  2.00000000e+00]]

Error Metrics for Y1:
MSE: 1.262177448353619e-29, MAE: 3.552713678800501e-15, RMSE: 3.552713678800501e-15, R2: 1.0, Adj R2: 1.0

Error Metrics for Y2:
MSE: 1.9721522630525295e-29, MAE: 4.440892098500626e-15, RMSE: 4.440892098500626e-15, R2: 1.0, Adj R2: 1.0

Predicted vs Actual Data:
x1=1 -> Actual Y1=2, Predicted Y1=2.000 | Actual Y2=3, Predicted Y2=3.000
x1=2 -> Actual Y1=4, Predicted Y1=4.000 | Actual Y2=5, Predicted Y2=5.000
x1=3 -> Actual Y1=6, Predicted Y1=6.000 | Actual Y2=7, Predicted Y2=7.000
