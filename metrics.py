import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils import validate_numeric_dataframe


# def compute_metrics(freq_data, metrics, programmers, output_path):
#     for metric in metrics:
#         if metric == "cosine":
#             similarity_df = compute_cosine_similarity(freq_data['unigram']['sparse_matrix'], programmers)
#             save_results(similarity_df, output_path, "cosine_similarity_unigram")
#         elif metric == "euclidean":
#             distance_df = compute_euclidean_distance(freq_data['unigram']['sparse_matrix'], programmers)
#             save_results(distance_df, output_path, "euclidean_distance_unigram")
#         elif metric == "pearson":
#             pearson_df = compute_pearson_correlation(freq_data['unigram']['matrix'], programmers)
#             save_results(pearson_df, output_path, "pearson_correlation_unigram")
#         elif metric == "jaccard":
#             jaccard_df = compute_jaccard_similarity(freq_data['unigram']['matrix'], programmers)
#             save_results(jaccard_df, output_path, "jaccard_similarity_unigram")


def compute_cosine_similarity(sparse_matrix, programmers):
    """
    Computes the cosine similarity between programmers based on their N-gram frequency vectors.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing cosine similarity scores between programmers.
    """
    # Transpose the matrix to have programmers as rows
    programmer_vectors = sparse_matrix.transpose()

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(programmer_vectors)

    # Create a DataFrame for better readability
    similarity_df = pd.DataFrame(similarity_matrix, index=programmers, columns=programmers)

    # Validate that the similarity_df contains only numeric data
    similarity_df = validate_numeric_dataframe(similarity_df, name="Cosine Similarity DataFrame")

    return similarity_df


def compute_euclidean_distance(sparse_matrix, programmers):
    """
    Computes the Euclidean distance between programmers based on their N-gram frequency vectors.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Euclidean distance scores between programmers.
    """
    # Transpose the matrix to have programmers as rows
    programmer_vectors = sparse_matrix.transpose()

    # Compute Euclidean distances
    distance_matrix = euclidean_distances(programmer_vectors)

    # Create a DataFrame for better readability
    distance_df = pd.DataFrame(distance_matrix, index=programmers, columns=programmers)

    # Validate that the distance_df contains only numeric data
    distance_df = validate_numeric_dataframe(distance_df, name="Euclidean Distance DataFrame")

    return distance_df


def compute_pearson_correlation(freq_matrix, programmers):
    """
    Computes the Pearson correlation coefficient between programmers based on their N-gram frequency vectors.

    Parameters:
        freq_matrix (pd.DataFrame): Dense frequency matrix with N-grams as rows and programmers as columns.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Pearson correlation coefficients between programmers.
    """
    correlation_df = freq_matrix.corr(method='pearson').fillna(0)
    return validate_numeric_dataframe(correlation_df, name="Pearson Correlation DataFrame")


def compute_jaccard_similarity(freq_matrix, programmers):
    """
    Computes the Jaccard similarity between programmers based on their N-gram presence.

    Parameters:
        freq_matrix (pd.DataFrame): Dense frequency matrix with N-grams as rows and programmers as columns.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Jaccard similarity scores between programmers.
    """
    binary_matrix = (freq_matrix > 0).astype(int)
    intersection = binary_matrix.T.dot(binary_matrix)
    union = binary_matrix.sum(axis=0) + binary_matrix.sum(axis=0).T - intersection
    jaccard_df = intersection.div(union).fillna(0)
    return validate_numeric_dataframe(jaccard_df, name="Jaccard Similarity DataFrame")


# def save_results(df, output_path, filename):
#     csv_path = f"{output_path}/{filename}.csv"
#     df.to_csv(csv_path)
#     logger.info(f"Saved {filename} to {csv_path}.")
