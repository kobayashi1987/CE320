import pandas as pd
from sklearn.cluster import KMeans


def cluster_programmers(sparse_matrix, programmers, output_path, n_clusters=2):
    """
    Clusters programmers based on their N-gram frequency vectors using K-Means.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.
        output_path (str): Path to save the clustering visualization.
        n_clusters (int): Number of clusters to form.

    Returns:
        pd.Series: Series mapping programmers to their assigned cluster.
    """
    # Transpose to have programmers as samples
    programmer_vectors = sparse_matrix.transpose()

    # Initialize K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit and predict clusters
    clusters = kmeans.fit_predict(programmer_vectors)

    # Create a Series mapping programmers to clusters
    cluster_series = pd.Series(clusters, index=programmers, name='Cluster')

    return cluster_series
