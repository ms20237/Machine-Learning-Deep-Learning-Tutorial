import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Reading the data from the .dat file
file_path = 'flutter.dat' 
with open(file_path, 'r') as file: 
    content = file.read()

# Create a StringIO object to simulate a file for pandas
data = StringIO(content)

# Reading the data into a DataFrame
df = pd.read_csv(data, delim_whitespace=True, header=None)

# Check the first few rows to understand the structure
print(df.head())
print(df.shape)  # Print the shape to check the number of columns
print(df.keys())



# Extracting input (X) and output (y) values
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column

# Plotting the data
plt.scatter(X, y)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Scatter plot of Input vs Output')
plt.show()


# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X_scaled)

# Get cluster labels
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(X_scaled, y, c=labels)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('K-means Clustering')
plt.show()



# Initialize K-medoids
initial_medoids = np.random.choice(len(X), 3, replace=False)
kmedoids_instance = kmedoids(X, initial_medoids)

# Run the clustering process and obtain the clusters
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx, cluster in enumerate(clusters):
    cluster_points = np.array(cluster)
    
    plt.scatter(X[cluster_points, 0], y[cluster_points], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('K-medoids Clustering')
plt.legend()
plt.show()


# Combine X and y into a single array
data_combined = np.hstack((X_scaled, y.reshape(-1, 1)))

# Perform hierarchical clustering
linkage_data = linkage(data_combined, method='ward', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_data)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()



# Initialize Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Fit the data to the clustering algorithm
clustering.fit(X)

# Get the cluster labels
labels = clustering.labels_

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx in range(3):
    cluster_points = X[labels == cluster_idx]
    plt.scatter(cluster_points[:, 0], y[labels == cluster_idx], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Agglomerative Clustering')
plt.legend()
plt.show()



# Initialize Agglomerative Clustering with the 'complete' linkage method
clustering = AgglomerativeClustering(n_clusters=3, linkage='complete')

# Fit the data to the clustering algorithm
clustering.fit(X)

# Get the cluster labels
labels = clustering.labels_

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx in range(3):
    cluster_points = X[labels == cluster_idx]
    plt.scatter(cluster_points[:, 0], y[labels == cluster_idx], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Divisive Clustering')
plt.legend()
plt.show()



# Initialize DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=5)

# Fit the data to the clustering algorithm
clustering.fit(X)

# Get the cluster labels
labels = clustering.labels_

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx in np.unique(labels):
    if cluster_idx == -1:
        # Noise points
        cluster_points = X[labels == -1]
        plt.scatter(cluster_points[:, 0], y[labels == -1], color='k', marker='x', label='Noise')
    else:
        # Cluster points
        cluster_points = X[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], y[labels == cluster_idx], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()




# Initialize OPTICS
clustering = OPTICS(min_samples=5)

# Fit the data to the clustering algorithm
clustering.fit(X)

# Get the cluster labels
labels = clustering.labels_

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx in np.unique(labels):
    if cluster_idx == -1:
        # Noise points
        cluster_points = X[labels == -1]
        plt.scatter(cluster_points[:, 0], y[labels == -1], color='k', marker='x', label='Noise')
    else:
        # Cluster points
        cluster_points = X[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], y[labels == cluster_idx], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('OPTICS Clustering')
plt.legend()
plt.show()




# Initialize GMM
gmm = GaussianMixture(n_components=3, covariance_type='full')

# Fit the data to the GMM
gmm.fit(X)

# Get the cluster labels
labels = gmm.predict(X)

# Visualize the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_idx in np.unique(labels):
    cluster_points = X[labels == cluster_idx]
    plt.scatter(cluster_points[:, 0], y[labels == cluster_idx], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Gaussian Mixture Model Clustering')
plt.legend()
plt.show()
