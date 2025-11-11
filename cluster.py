import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

healthDataSet = pd.read_csv("health_lifestyle_modified.csv")

features = ['bmi_minmax', 'calories_consumed_minmax']

scaler = StandardScaler()
chosenDatasetColumnValues = healthDataSet[features].values
chosenDatasetColumnValues_scaled = scaler.fit_transform(chosenDatasetColumnValues)

print(chosenDatasetColumnValues.shape)
print(len(features))

# Visualize the dataset
plt.figure(figsize=(10, 6))
plt.scatter(chosenDatasetColumnValues[:, 0], chosenDatasetColumnValues[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Function to plot k-distance graph
def plot_k_distance_graph(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.show()
# Plot k-distance graph
plot_k_distance_graph(chosenDatasetColumnValues, k=5) #0.09 epszilon

# Perform DBSCAN clustering
epsilon = 0.005  # Chosen based on k-distance graph
min_samples = 5  # 2 * num_features (2D data)
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(chosenDatasetColumnValues)

# Visualize the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(chosenDatasetColumnValues[:, 0], chosenDatasetColumnValues[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Print number of clusters and noise points
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')
