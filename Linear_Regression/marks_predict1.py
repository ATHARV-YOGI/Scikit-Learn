from sklearn.linear_model import LinearRegression

x = [[1],[2],[3],[4],[5]]
y = [10,20,30,40,50]

model = LinearRegression()
model.fit(x,y)

hours = float(input("enter how many hours you studied = "))

predicted_marks = model.predict([[hours]])

print(f"based on your hours (hours) you may score around {predicted_marks}")