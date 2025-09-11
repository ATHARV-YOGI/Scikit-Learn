import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("penguins.csv")

X = data[['hours']]
y = data['score']

model = LinearRegression()
model.fit(X,y)

predicted_score = model.predict(X)

mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

print("mean absolute error (MAE): ", mae)
print("mean square error (MSE): ", mse)
print("root mean squared error (RMSE): ", mae)

new_hour = float(input("enter a hour = "))
new_pred = model.predict([[new_hour]])
print(f"prediction for {new_hour} is score = {new_pred}")