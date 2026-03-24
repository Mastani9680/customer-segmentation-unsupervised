from sklearn.cluster import AgglomerativeClustering

def run_hierarchical(X_scaled):
    model = AgglomerativeClustering(n_clusters=4)
    return model.fit_predict(X_scaled)