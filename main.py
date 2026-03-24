
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")

print("Columns:", df.columns)

# -----------------------------
# DATE CONVERSION
# -----------------------------
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# -----------------------------
# RFM FEATURE ENGINEERING
# -----------------------------
reference_date = df['TransactionDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (reference_date - x.max()).days,
    'CustomerID': 'count',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\nRFM Sample:")
print(rfm.head())

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
k_labels = kmeans.fit_predict(X_pca)

gmm = GaussianMixture(n_components=4, random_state=42)
g_labels = gmm.fit_predict(X_pca)

# -----------------------------
# EVALUATION
# -----------------------------
k_score = silhouette_score(X_pca, k_labels)
g_score = silhouette_score(X_pca, g_labels)

print("\n📊 Silhouette Scores:")
print("KMeans:", k_score)
print("GMM:", g_score)

# -----------------------------
# CREATE RESULTS FOLDER
# -----------------------------
os.makedirs("results/cluster_plots", exist_ok=True)

# -----------------------------
# KMEANS PLOT
# -----------------------------
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels)
plt.title("KMeans Clusters")
plt.savefig("kmeans.png")
plt.close()

# -----------------------------
# GMM PLOT
# -----------------------------
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=g_labels)
plt.title("GMM Clusters")
plt.savefig("gmm.png")
plt.close()

# -----------------------------
# SAVE OUTPUT
# -----------------------------
rfm['Cluster'] = k_labels
rfm.to_csv("customer_segments.csv")

# -----------------------------
# SUMMARY
# -----------------------------
print("\n📊 Cluster Summary:")
print(rfm.groupby("Cluster").mean())

print("\n✅ Project executed successfully!")