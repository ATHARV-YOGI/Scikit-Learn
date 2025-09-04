from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("Sample_Data.csv")

# Clean column names (remove spaces, make consistent)
df.columns = df.columns.str.strip()

df_label = df.copy()
le = LabelEncoder()

print("Columns after cleaning:", df_label.columns.tolist())  # Debug

df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender'])
df_label['Passed_Encoded'] = le.fit_transform(df_label['Passed'])

print('\nLabel Encoded Data')
print(df_label[['Name','Gender','Gender_Encoded','Passed','Passed_Encoded']])
