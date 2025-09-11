import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("stroke.csv")

x = data[['avg_glucose_level']]
y = data['age']

model = LinearRegression()

model.fit(x,y)

predicted_scores = model.predict(x)

mae = mean_absolute_error(y, predicted_scores)
mse = mean_squared_error(y, predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_scores)

print("mean absolute error (mae): ", round(mae, 2))
print("mean squared error (mse): ", round(mse, 2))
print("root mean squared error (rmse): ", round(rmse, 2))
print("R^2 score (model accuracy): ", round(r2, 4))

plt.figure(figsize=(10, 6))
plt.hist(data["age"], bins=30, color='skyblue', edgecolor='black')
plt.title("distribution of final exam scores")
plt.xlabel("final age")
plt.ylabel("avg glucose levvel")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(x, y,color = 'blue', label='actual score')
plt.plot(x,predicted_scores,color='red',label="predicted scores(regression line)")
plt.title("distribution of final exam scores")
plt.xlabel("final age")
plt.ylabel("avg glucose levvel")
plt.grid(True)
plt.show()


new_hours = 9
predicted_new_score = model.predict([[new_hours]])
print(f"predicted final score for {new_hours} hours is {predicted_new_score} score")