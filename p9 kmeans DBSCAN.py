from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
data = load_wine()
X = StandardScaler().fit_transform(data.data)
# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X)
# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
# Plot K-means
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='Set1')
plt.title('K-Means Clustering')
plt.show()
# Plot DBSCAN
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap='Set2')
plt.title('DBSCAN Clustering')
plt.show()