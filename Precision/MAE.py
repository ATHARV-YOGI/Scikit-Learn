from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

real_scores = [90,60,80,100]

predicted_scores = [85,70,70,95]

mae = mean_absolute_error(real_scores,predicted_scores)

mse = mean_squared_error(real_scores,predicted_scores)

rmse = np.sqrt(mse)

print("MAE: on average off by: ",mae)
print("MSE: squared mistake value: ",mse)
print("RMSE: final realistic error: ",rmse)