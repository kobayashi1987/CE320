import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils import validate_numeric_dataframe


def compute_similarity_metrics(sparse_matrix, programmers, similarity_func, metric_name):
    """
    Computes a similarity metric (e.g., cosine similarity) and returns a DataFrame.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix of programmer data.
        programmers (list): List of programmer identifiers.
        similarity_func (function): Similarity function (e.g., cosine_similarity).
        metric_name (str): Name of the similarity metric.

    Returns:
        pd.DataFrame: DataFrame containing the similarity scores.
    """

    # Convert sparse matrix to dense if necessary
    if isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.toarray()

    # Validate that the number of rows matches the number of programmers
    if sparse_matrix.shape[0] != len(programmers):
        raise ValueError(
            f"Mismatch: similarity matrix rows {sparse_matrix.shape[0]} do not match "
            f"programmers list size {len(programmers)}"
        )

    # Compute similarity or distance
    similarity_matrix = similarity_func(sparse_matrix)

    # Create DataFrame
    return pd.DataFrame(similarity_matrix, index=programmers, columns=programmers)


def compute_cosine_similarity(sparse_matrix, programmers):
    """
    Computes the cosine similarity between programmers based on their N-gram frequency vectors.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing cosine similarity scores between programmers.
    """

    # Transpose the sparse matrix to have programmers as rows
    if isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.transpose()

    # Convert sparse matrix to dense if necessary
    dense_matrix = sparse_matrix.toarray()

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(dense_matrix)

    # Create DataFrame with programmer names as indices and columns
    return pd.DataFrame(similarity_matrix, index=programmers, columns=programmers)


def compute_euclidean_distance(sparse_matrix, programmers):
    """
    Computes the Euclidean distance between programmers based on their N-gram frequency vectors.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Euclidean distance scores between programmers.
    """

    return compute_similarity_metrics(sparse_matrix, programmers, euclidean_distances, "Euclidean Distance")


def compute_pearson_correlation(freq_matrix, programmers):
    """
    Computes the Pearson correlation coefficient between programmers based on their N-gram frequency vectors.

    Parameters:
        freq_matrix (pd.DataFrame or csr_matrix): Dense frequency matrix where rows represent programmers.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Pearson correlation coefficients between programmers.
    """
    # Convert sparse matrix to dense DataFrame if necessary
    if isinstance(freq_matrix, (csr_matrix, csc_matrix)):
        freq_matrix = pd.DataFrame(
            freq_matrix.toarray(),
            index=programmers
        )

    # Transpose the matrix so rows represent programmers
    freq_matrix = freq_matrix.T

    # Compute Pearson correlation on rows (programmers)
    correlation_df = freq_matrix.corr(method='pearson').fillna(0)
    return correlation_df


def compute_jaccard_similarity(freq_matrix, programmers):
    """
    Computes the Jaccard similarity between programmers based on their N-gram presence.

    Parameters:
        freq_matrix (csr_matrix or pd.DataFrame): Sparse or dense frequency matrix.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Jaccard similarity scores between programmers.
    """
    # Convert sparse matrix to a dense DataFrame if necessary
    if isinstance(freq_matrix, (csr_matrix, csc_matrix)):
        freq_matrix = pd.DataFrame(freq_matrix.toarray(), index=programmers)

    # Binarize the matrix (presence/absence of N-grams)
    binary_matrix = (freq_matrix > 0).astype(int)

    # Compute intersection and union
    intersection = binary_matrix.dot(binary_matrix.T)
    union = binary_matrix.sum(axis=1).values[:, None] + binary_matrix.sum(axis=1).values - intersection

    # Calculate Jaccard similarity
    jaccard_df = pd.DataFrame(intersection / union, index=programmers, columns=programmers).fillna(0)
    return validate_numeric_dataframe(jaccard_df, name="Jaccard Similarity")

