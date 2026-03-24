from sklearn.cluster import KMeans

def run_kmeans(X_scaled):
    model = KMeans(n_clusters=4, random_state=42)
    return model.fit_predict(X_scaled)