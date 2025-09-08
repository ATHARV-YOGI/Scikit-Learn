from sklearn.linear_model import LogisticRegression

x = [[1],[2],[3],[4],[5]]
y = [0,0,1,1,1]

model = LogisticRegression()

model.fit(x,y)

hours = float(input("enter how many hours you studies = "))

result = model.predict([[hours]])[0]

if result == 1:
    print(f"based on hours {hours}, you are likely to PASS")
else:
    print(f"based on hours {hours}, you are likely to FAIL")