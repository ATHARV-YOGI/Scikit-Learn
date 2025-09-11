from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# true ans (what actually happened)
y_true = [1,0,1,1,0,1,0]

# model's predictions (what it guessed)
y_pred = [1,0,1,0,0,1,1]

print("accuracy: ", accuracy_score(y_true, y_pred))
print("accuracy: ", precision_score(y_true, y_pred))
print("accuracy: ", recall_score(y_true, y_pred))
print("accuracy: ", f1_score(y_true, y_pred))