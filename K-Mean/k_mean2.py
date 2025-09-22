import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'Age':[12,23,34,56,67,78],
    'Income':[2222,3333,4444,6666,777,8888],
    'Spending':[23,34,54,23,23,54],
    'Saving':[1000,5000,4000,3444,2444,5444]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_result, columns=['PCA1','PCA2'])

explained_variance = pca.explained_variance_ratio_
print("variance captured by each PCA component")
print(np.round(explained_variance * 100, 2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], color='black', s=2)
plt.title("pca projection (2d view)")
plt.xlabel('pca1 main pattern')
plt.ylabel('pca2 minor pattern')
plt.grid(True)
plt.show()