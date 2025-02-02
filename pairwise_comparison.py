import os
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
import logging
from datetime import datetime

# Define project root path
project_root = "/Users/jack/Desktop/project/pycharm/ce320"

# Define input and output paths
frequency_dir = os.path.join(project_root, "output", "frequency_matrices")
similarity_metrics_dir = os.path.join(project_root, "output", "similarity_metrics")
visualizations_dir = os.path.join(project_root, "output", "visualizations")
logs_dir = os.path.join(project_root, "output", "logs")

input_path = os.path.join(project_root, "input")

# Ensure output directories exist
os.makedirs(similarity_metrics_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(logs_dir, f'pairwise_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)


def read_frequency_matrix(file_path):
    """
    Reads a frequency matrix CSV into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Frequency matrix with N-grams as rows and programmers as columns.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        logger.info(f"Successfully read frequency matrix from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading frequency matrix from {file_path}: {e}")
        return None


def compute_cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.

    Parameters:
        vec1 (pd.Series or np.array): First vector.
        vec2 (pd.Series or np.array): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    if not (vec1.any() and vec2.any()):
        return 0.0
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity


def compute_euclidean_distance(vec1, vec2):
    """
    Computes the Euclidean distance between two vectors.

    Parameters:
        vec1 (pd.Series or np.array): First vector.
        vec2 (pd.Series or np.array): Second vector.

    Returns:
        float: Euclidean distance score.
    """
    distance = euclidean_distances([vec1], [vec2])[0][0]
    return distance


def compute_pearson_correlation(vec1, vec2):
    """
    Computes the Pearson correlation coefficient between two vectors.

    Parameters:
        vec1 (pd.Series or np.array): First vector.
        vec2 (pd.Series or np.array): Second vector.

    Returns:
        float: Pearson correlation coefficient.
    """
    if len(vec1) != len(vec2):
        logger.warning("Vectors have different lengths for Pearson correlation.")
        return 0.0
    corr, _ = pearsonr(vec1, vec2)
    if pd.isna(corr):
        return 0.0
    return corr


def compute_jaccard_similarity(vec1, vec2):
    """
    Computes the Jaccard similarity between two binary vectors.

    Parameters:
        vec1 (pd.Series or np.array): First binary vector.
        vec2 (pd.Series or np.array): Second binary vector.

    Returns:
        float: Jaccard similarity score.
    """
    intersection = ((vec1 > 0) & (vec2 > 0)).sum()
    union = ((vec1 > 0) | (vec2 > 0)).sum()
    if union == 0:
        return 0.0
    return intersection / union


def perform_pairwise_comparison(unigram_df, bigram_df, programmers, output_path):
    """
    Performs pairwise comparison of programmers based on unigram and bigram frequency matrices.

    Parameters:
        unigram_df (pd.DataFrame): Unigram frequency matrix.
        bigram_df (pd.DataFrame): Bigram frequency matrix.
        programmers (list): List of programmer names.
        output_path (str): Path to save the pairwise comparison CSV.
    """
    pairwise_results = []

    # Generate all unique programmer pairs
    programmer_pairs = list(itertools.combinations(programmers, 2))

    logger.info(f"Total programmer pairs to compare: {len(programmer_pairs)}")

    for prog1, prog2 in programmer_pairs:
        logger.info(f"Comparing {prog1} and {prog2}")

        # Extract unigram frequency vectors
        uni_vec1 = unigram_df[prog1]
        uni_vec2 = unigram_df[prog2]

        # Extract bigram frequency vectors
        bi_vec1 = bigram_df[prog1]
        bi_vec2 = bigram_df[prog2]

        # Compute similarity metrics for unigrams
        cosine_uni = compute_cosine_similarity(uni_vec1, uni_vec2)
        euclidean_uni = compute_euclidean_distance(uni_vec1, uni_vec2)
        pearson_uni = compute_pearson_correlation(uni_vec1, uni_vec2)
        jaccard_uni = compute_jaccard_similarity(uni_vec1, uni_vec2)

        # Compute similarity metrics for bigrams
        cosine_bi = compute_cosine_similarity(bi_vec1, bi_vec2)
        euclidean_bi = compute_euclidean_distance(bi_vec1, bi_vec2)
        pearson_bi = compute_pearson_correlation(bi_vec1, bi_vec2)
        jaccard_bi = compute_jaccard_similarity(bi_vec1, bi_vec2)

        # Append the results
        pairwise_results.append({
            'Programmer1': prog1,
            'Programmer2': prog2,
            'Cosine_Similarity_Unigrams': cosine_uni,
            'Cosine_Similarity_Bigrams': cosine_bi,
            'Euclidean_Distance_Unigrams': euclidean_uni,
            'Euclidean_Distance_Bigrams': euclidean_bi,
            'Pearson_Correlation_Unigrams': pearson_uni,
            'Pearson_Correlation_Bigrams': pearson_bi,
            'Jaccard_Similarity_Unigrams': jaccard_uni,
            'Jaccard_Similarity_Bigrams': jaccard_bi
        })

    # Create a DataFrame from the results
    pairwise_df = pd.DataFrame(pairwise_results)

    # Save the results to CSV to both output and input directories
    # for input directories, the file will be used in the next step
    output_file_output = os.path.join(output_path, 'pairwise_comparison.csv')
    output_file_input = os.path.join(input_path, 'pairwise_comparison.csv')
    pairwise_df.to_csv(output_file_input, index=False)
    pairwise_df.to_csv(output_file_output, index=False)
    logger.info(f"Pairwise comparison results saved to {output_file_output} and {output_file_input}")

    # Optionally, print the DataFrame
    print("\nPairwise Comparison Results:")
    print(pairwise_df)


def main():
    # Define file paths
    unigram_file = os.path.join(frequency_dir, 'unigram_frequency_matrix.csv')
    bigram_file = os.path.join(frequency_dir, 'bigram_frequency_matrix.csv')

    # Read frequency matrices
    unigram_df = read_frequency_matrix(unigram_file)
    bigram_df = read_frequency_matrix(bigram_file)

    if unigram_df is None or bigram_df is None:
        logger.error("Failed to read frequency matrices. Exiting.")
        return

    # Ensure that the columns (programmers) are the same in both matrices
    programmers_unigram = set(unigram_df.columns)
    programmers_bigram = set(bigram_df.columns)

    if programmers_unigram != programmers_bigram:
        logger.error("Mismatch in programmer lists between unigram and bigram frequency matrices.")
        return

    programmers = sorted(list(programmers_unigram))

    logger.info(f"Programmers found: {programmers}")

    # Perform pairwise comparison
    perform_pairwise_comparison(unigram_df, bigram_df, programmers, similarity_metrics_dir)


if __name__ == "__main__":
    main()