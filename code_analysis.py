import os
import tokenize
import io
from token import tok_name
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import logging
from datetime import datetime


# Configure logging
def setup_logging(output_path):
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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


class NGramGenerator:
    def __init__(self, code, desired_types=None):
        """
        Initializes the NGramGenerator with Python code.

        Parameters:
            code (str): The Python source code to analyze.
            desired_types (set, optional): A set of token types to keep.
                Defaults to {'NAME', 'NUMBER', 'STRING', 'OP'}.
        """
        self.code = code
        self.desired_types = desired_types if desired_types else {'NAME', 'NUMBER', 'STRING', 'OP'}
        self.tokens = self.tokenize_code()
        self.filtered_tokens = list(self.filter_tokens())  # Convert generator to list

    def tokenize_code(self):
        """
        Tokenizes the Python code using the tokenize module.

        Returns:
            generator: A generator of token tuples (token_type, token_string).
        """
        code_bytes = io.BytesIO(self.code.encode('utf-8'))
        try:
            for tok in tokenize.tokenize(code_bytes.readline):
                if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                    continue
                token_type = tok_name.get(tok.type, tok.type)
                token_string = tok.string
                yield (token_type, token_string)
        except tokenize.TokenError as e:
            logger.error(f"Tokenization error: {e}")
            return

    def filter_tokens(self):
        """
        Filters tokens based on desired types.

        Returns:
            generator: A generator of filtered token tuples.
        """
        return (token for token in self.tokens if token[0] in self.desired_types)

    def generate_ngrams_generator(self, n=2):
        """
        Generates N-grams using a generator to handle large datasets efficiently.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Yields:
            str: An N-gram string.
        """
        if n <= 0:
            raise ValueError("N must be a positive integer.")
        token_strings = [token[1] for token in self.filtered_tokens]
        if n > len(token_strings):
            logger.warning(f"N={n} is greater than the number of tokens ({len(token_strings)}). No N-grams generated.")
            return
        for i in range(len(token_strings) - n + 1):
            ngram = ' '.join(token_strings[i:i + n])
            yield ngram

    def count_ngram_frequencies_counter(self, n=2):
        """
        Counts the frequency of each N-gram using collections.Counter.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        ngram_gen = self.generate_ngrams_generator(n)
        return Counter(ngram_gen)


def create_mappings(freq_matrix):
    """
    Creates mappings from N-grams to indices and programmers to indices.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix with N-grams as rows and programmers as columns.

    Returns:
        tuple: (ngram_to_index, programmer_to_index)
    """
    ngram_to_index = {ngram: idx for idx, ngram in enumerate(freq_matrix.index)}
    programmer_to_index = {programmer: idx for idx, programmer in enumerate(freq_matrix.columns)}
    return ngram_to_index, programmer_to_index


def convert_to_sparse_matrix(freq_matrix):
    """
    Converts a Pandas DataFrame to a SciPy CSR sparse matrix.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to convert.

    Returns:
        csr_matrix: The converted sparse matrix.
    """
    return csr_matrix(freq_matrix.values)


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
    correlation_df = pd.DataFrame(index=programmers, columns=programmers)

    for prog1 in programmers:
        for prog2 in programmers:
            if prog1 == prog2:
                correlation_df.at[prog1, prog2] = 1.0
            else:
                # Compute Pearson correlation
                corr, _ = pearsonr(freq_matrix[prog1], freq_matrix[prog2])
                # Handle cases where correlation is undefined
                if pd.isna(corr):
                    correlation_df.at[prog1, prog2] = 0.0
                else:
                    correlation_df.at[prog1, prog2] = corr

    # Validate that the correlation_df contains only numeric data
    correlation_df = validate_numeric_dataframe(correlation_df, name="Pearson Correlation DataFrame")

    return correlation_df


def compute_jaccard_similarity(freq_matrix, programmers):
    """
    Computes the Jaccard similarity between programmers based on their N-gram presence.

    Parameters:
        freq_matrix (pd.DataFrame): Dense frequency matrix with N-grams as rows and programmers as columns.
        programmers (list): List of programmer identifiers.

    Returns:
        pd.DataFrame: DataFrame containing Jaccard similarity scores between programmers.
    """
    # Convert frequency counts to binary (presence/absence) without using applymap
    binary_matrix = (freq_matrix > 0).astype(int)

    jaccard_df = pd.DataFrame(index=programmers, columns=programmers)

    for prog1 in programmers:
        for prog2 in programmers:
            if prog1 == prog2:
                jaccard_df.at[prog1, prog2] = 1.0
            else:
                intersection = (binary_matrix[prog1] & binary_matrix[prog2]).sum()
                union = (binary_matrix[prog1] | binary_matrix[prog2]).sum()
                jaccard = intersection / union if union != 0 else 0
                jaccard_df.at[prog1, prog2] = jaccard

    # Validate that the jaccard_df contains only numeric data
    jaccard_df = validate_numeric_dataframe(jaccard_df, name="Jaccard Similarity DataFrame")

    return jaccard_df


def validate_numeric_dataframe(df, name="DataFrame"):
    """
    Validates that a DataFrame contains only numeric data types.
    If not, attempts to convert to numeric, coercing errors to NaN and filling them with zero.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        name (str): The name of the DataFrame for informative messages.

    Returns:
        pd.DataFrame: A validated numeric DataFrame.
    """
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        logger.warning(f"{name} contains non-numeric data types. Attempting to convert...")
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        logger.info(f"{name} successfully converted to numeric types.")
    else:
        logger.info(f"{name} contains only numeric data types.")
    return df


def visualize_similarity_matrix(similarity_df, output_path, title="Similarity Matrix", cmap='YlGnBu', fmt=".2f"):
    """
    Visualizes the similarity or distance matrix using a heatmap.

    Parameters:
        similarity_df (pd.DataFrame): DataFrame containing similarity or distance scores.
        output_path (str): Path to save the heatmap image.
        title (str): Title of the heatmap.
        cmap (str): Colormap to use for the heatmap.
        fmt (str): String formatting code.
    """
    # Validate that the similarity_df contains only numeric data
    similarity_df = validate_numeric_dataframe(similarity_df, name="Similarity DataFrame")

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, fmt=fmt, cmap=cmap, square=True)
    plt.title(title)
    plt.xlabel('Programmers')
    plt.ylabel('Programmers')
    plt.tight_layout()

    # Save the heatmap
    heatmap_filename = f"{title.replace(' ', '_').lower()}.png"
    heatmap_path = os.path.join(output_path, 'visualizations', heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved heatmap: {heatmap_filename}")


def normalize_frequency_matrix(freq_matrix):
    """
    Normalizes the frequency matrix using Term Frequency (TF).
    Handles division by zero by replacing NaNs with zeros.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to normalize.

    Returns:
        pd.DataFrame: The normalized frequency matrix with no NaN values.
    """
    normalized = freq_matrix.div(freq_matrix.sum(axis=0), axis=1)
    normalized = normalized.fillna(0)  # Replace NaNs resulting from division by zero
    return normalized


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


def visualize_clustering(clusters, output_path, title="Programmer Clusters"):
    """
    Visualizes clustering results using a bar plot.

    Parameters:
        clusters (pd.Series): Series mapping programmers to clusters.
        output_path (str): Path to save the clustering visualization.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=clusters)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Programmers')
    plt.tight_layout()

    # Save the clustering plot
    clustering_filename = f"{title.replace(' ', '_').lower()}.png"
    clustering_path = os.path.join(output_path, 'visualizations', clustering_filename)
    plt.savefig(clustering_path)
    plt.close()
    logger.info(f"Saved clustering visualization: {clustering_filename}")


def create_frequency_matrices(unigram_freqs, bigram_freqs, programmers, output_path):
    """
    Creates unigram and bigram frequency matrices and saves them as CSV files.

    Parameters:
        unigram_freqs (dict): Dictionary mapping programmers to their unigram frequency Counters.
        bigram_freqs (dict): Dictionary mapping programmers to their bigram frequency Counters.
        programmers (list): List of programmer identifiers.
        output_path (str): Path to save the frequency matrices.

    Returns:
        dict: Contains unigram and bigram frequency matrices and their sparse representations.
    """
    # Create a set of all unique unigrams across all programmers
    all_unigrams = set()
    for freq in unigram_freqs.values():
        all_unigrams.update(freq.keys())

    # Convert the set to a sorted list
    all_unigrams_sorted = sorted(all_unigrams)

    # Initialize the unigram frequency matrix with a sorted list as index
    unigram_matrix = pd.DataFrame(0, index=all_unigrams_sorted, columns=programmers)

    # Populate the unigram frequency matrix
    for programmer, freq in unigram_freqs.items():
        for unigram, count in freq.items():
            unigram_matrix.at[unigram, programmer] = count

    logger.info("Unigram Frequency Matrix:")
    logger.info(unigram_matrix)

    # Similarly, create the bigram frequency matrix
    all_bigrams = set()
    for freq in bigram_freqs.values():
        all_bigrams.update(freq.keys())

    # Convert the set to a sorted list
    all_bigrams_sorted = sorted(all_bigrams)

    # Initialize the bigram frequency matrix with a sorted list as index
    bigram_matrix = pd.DataFrame(0, index=all_bigrams_sorted, columns=programmers)

    # Populate the bigram frequency matrix
    for programmer, freq in bigram_freqs.items():
        for bigram, count in freq.items():
            bigram_matrix.at[bigram, programmer] = count

    logger.info("Bigram Frequency Matrix:")
    logger.info(bigram_matrix)

    # Create mappings
    unigram_ngram_to_index, programmer_to_index_uni = create_mappings(unigram_matrix)
    bigram_ngram_to_index, programmer_to_index_bi = create_mappings(bigram_matrix)

    # Convert to sparse matrices
    unigram_sparse = convert_to_sparse_matrix(unigram_matrix)
    bigram_sparse = convert_to_sparse_matrix(bigram_matrix)

    # Save frequency matrices as CSV
    frequency_dir = os.path.join(output_path, 'frequency_matrices')
    os.makedirs(frequency_dir, exist_ok=True)
    unigram_matrix.to_csv(os.path.join(frequency_dir, 'unigram_frequency_matrix.csv'))
    bigram_matrix.to_csv(os.path.join(frequency_dir, 'bigram_frequency_matrix.csv'))
    logger.info("Saved frequency matrices as CSV files.")

    # Visualize the matrices
    visualize_frequency_matrix(unigram_matrix, output_path, title="Unigram Frequency Matrix", top_n=20)
    visualize_frequency_matrix(bigram_matrix, output_path, title="Bigram Frequency Matrix", top_n=20)

    # Return all necessary components for further use
    return {
        'unigram': {
            'matrix': unigram_matrix,
            'sparse_matrix': unigram_sparse,
            'ngram_to_index': unigram_ngram_to_index,
            'programmer_to_index': programmer_to_index_uni
        },
        'bigram': {
            'matrix': bigram_matrix,
            'sparse_matrix': bigram_sparse,
            'ngram_to_index': bigram_ngram_to_index,
            'programmer_to_index': programmer_to_index_bi
        }
    }


def visualize_frequency_matrix(freq_matrix, output_path, title="Frequency Matrix", top_n=20):
    """
    Visualizes the top N N-grams in the frequency matrix using a heatmap.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to visualize.
        output_path (str): Path to save the heatmap image.
        title (str): The title of the plot.
        top_n (int): The number of top N-grams to display.
    """
    if freq_matrix.empty:
        logger.warning("Frequency matrix is empty. Nothing to plot.")
        return

    # Sum frequencies across programmers to find top N-grams
    top_ngrams = freq_matrix.sum(axis=1).sort_values(ascending=False).head(top_n).index
    top_freq_matrix = freq_matrix.loc[top_ngrams]

    # Validate that the top_freq_matrix contains only numeric data
    top_freq_matrix = validate_numeric_dataframe(top_freq_matrix, name="Top Frequency Matrix")

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_freq_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Programmers')
    plt.ylabel('N-grams')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the heatmap
    heatmap_filename = f"{title.replace(' ', '_').lower()}.png"
    heatmap_path = os.path.join(output_path, 'visualizations', heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved heatmap: {heatmap_filename}")


def main():
    # Define project root path
    project_root = "/Users/jack/Desktop/project/pycharm/ce320"

    # Define input and output paths
    input_path = os.path.join(project_root, "input")
    output_path = os.path.join(project_root, "output")

    # Setup logging
    setup_logging(output_path)
    logger.info("Starting code analysis...")

    # Validate input directory
    if not os.path.isdir(input_path):
        logger.error(f"Input path '{input_path}' is not a valid directory.")
        return

    # Create necessary output subdirectories
    os.makedirs(os.path.join(output_path, 'frequency_matrices'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'similarity_metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)

    # Initialize dictionaries to hold N-gram frequencies
    unigram_freqs = {}
    bigram_freqs = {}

    # Iterate over each programmer's directory
    programmers = []
    for programmer_dir in os.listdir(input_path):
        programmer_path = os.path.join(input_path, programmer_dir)
        if os.path.isdir(programmer_path):
            programmer = programmer_dir
            programmers.append(programmer)
            logger.info(f"Processing code for {programmer}...")
            code_combined = ""

            # Read all code files for the programmer
            for root, _, files in os.walk(programmer_path):
                for file in files:
                    if file.endswith('.py'):  # Modify this if analyzing other languages
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code = f.read()
                                code_combined += code + "\n"
                        except Exception as e:
                            logger.error(f"Failed to read file '{file_path}': {e}")

            # Initialize NGramGenerator
            generator = NGramGenerator(code_combined)

            # Count unigrams
            uni_freq = generator.count_ngram_frequencies_counter(n=1)
            unigram_freqs[programmer] = uni_freq

            # Count bigrams
            bi_freq = generator.count_ngram_frequencies_counter(n=2)
            bigram_freqs[programmer] = bi_freq
        else:
            logger.warning(f"Skipping non-directory item in input path: {programmer_dir}")

    if not programmers:
        logger.error("No valid programmer directories found in the input path.")
        return

    # Create frequency matrices and convert to sparse format
    frequency_data = create_frequency_matrices(unigram_freqs, bigram_freqs, programmers, output_path)

    # Access the sparse matrices and mappings if needed
    unigram_sparse = frequency_data['unigram']['sparse_matrix']
    bigram_sparse = frequency_data['bigram']['sparse_matrix']
    unigram_mappings = frequency_data['unigram']['ngram_to_index']
    bigram_mappings = frequency_data['bigram']['ngram_to_index']

    # Example: Accessing a specific frequency
    # Print the frequency of the unigram 'add' for 'Alice' if exists
    if 'add' in frequency_data['unigram']['matrix'].index and 'Alice' in frequency_data['unigram']['matrix'].columns:
        print("\nFrequency of unigram 'add' for Alice:", frequency_data['unigram']['matrix'].at['add', 'Alice'])
    else:
        logger.warning("The unigram 'add' for 'Alice' does not exist in the frequency matrix.")

    # Compute and save Cosine Similarity between programmers based on unigrams
    similarity_unigram_df = compute_cosine_similarity(unigram_sparse, programmers)
    similarity_unigram_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'cosine_similarity_unigram.csv'))
    logger.info("Saved cosine similarity (unigrams) as CSV.")
    print("\nCosine Similarity between Programmers (Unigrams):")
    print(similarity_unigram_df)

    # Visualize the similarity matrix for unigrams
    visualize_similarity_matrix(similarity_unigram_df, output_path,
                                title="Cosine Similarity between Programmers (Unigrams)")

    # Compute and save Cosine Similarity between programmers based on bigrams
    similarity_bigram_df = compute_cosine_similarity(bigram_sparse, programmers)
    similarity_bigram_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'cosine_similarity_bigram.csv'))
    logger.info("Saved cosine similarity (bigrams) as CSV.")
    print("\nCosine Similarity between Programmers (Bigrams):")
    print(similarity_bigram_df)

    # Visualize the similarity matrix for bigrams
    visualize_similarity_matrix(similarity_bigram_df, output_path,
                                title="Cosine Similarity between Programmers (Bigrams)")

    # Compute and save Euclidean Distance between programmers based on unigrams
    euclidean_unigram_df = compute_euclidean_distance(unigram_sparse, programmers)
    euclidean_unigram_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'euclidean_distance_unigram.csv'))
    logger.info("Saved Euclidean distance (unigrams) as CSV.")
    print("\nEuclidean Distance between Programmers (Unigrams):")
    print(euclidean_unigram_df)

    # Visualize the Euclidean distance matrix for unigrams
    visualize_similarity_matrix(euclidean_unigram_df, output_path,
                                title="Euclidean Distance between Programmers (Unigrams)", cmap='Reds', fmt=".2f")

    # Compute and save Pearson Correlation between programmers based on unigrams
    pearson_unigram_df = compute_pearson_correlation(frequency_data['unigram']['matrix'], programmers)
    pearson_unigram_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'pearson_correlation_unigram.csv'))
    logger.info("Saved Pearson correlation (unigrams) as CSV.")
    print("\nPearson Correlation between Programmers (Unigrams):")
    print(pearson_unigram_df)

    # Visualize the Pearson correlation matrix for unigrams
    visualize_similarity_matrix(pearson_unigram_df, output_path,
                                title="Pearson Correlation between Programmers (Unigrams)")

    # Compute and save Jaccard Similarity between programmers based on unigrams
    jaccard_unigram_df = compute_jaccard_similarity(frequency_data['unigram']['matrix'], programmers)
    jaccard_unigram_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'jaccard_similarity_unigram.csv'))
    logger.info("Saved Jaccard similarity (unigrams) as CSV.")
    print("\nJaccard Similarity between Programmers (Unigrams):")
    print(jaccard_unigram_df)

    # Visualize the Jaccard similarity matrix for unigrams
    visualize_similarity_matrix(jaccard_unigram_df, output_path,
                                title="Jaccard Similarity between Programmers (Unigrams)")

    # Optional: Normalize frequency matrices
    normalized_unigram = normalize_frequency_matrix(frequency_data['unigram']['matrix'])
    normalized_unigram.to_csv(
        os.path.join(output_path, 'frequency_matrices', 'normalized_unigram_frequency_matrix.csv'))
    logger.info("Saved normalized unigram frequency matrix as CSV.")
    print("\nNormalized Unigram Frequency Matrix:")
    print(normalized_unigram)

    normalized_bigram = normalize_frequency_matrix(frequency_data['bigram']['matrix'])
    normalized_bigram.to_csv(os.path.join(output_path, 'frequency_matrices', 'normalized_bigram_frequency_matrix.csv'))
    logger.info("Saved normalized bigram frequency matrix as CSV.")
    print("\nNormalized Bigram Frequency Matrix:")
    print(normalized_bigram)

    # Compute and save Cosine Similarity on Normalized Unigrams
    similarity_unigram_norm_df = compute_cosine_similarity(convert_to_sparse_matrix(normalized_unigram), programmers)
    similarity_unigram_norm_df.to_csv(
        os.path.join(output_path, 'similarity_metrics', 'cosine_similarity_normalized_unigram.csv'))
    logger.info("Saved cosine similarity (normalized unigrams) as CSV.")
    print("\nCosine Similarity between Programmers (Normalized Unigrams):")
    print(similarity_unigram_norm_df)

    # Visualize the cosine similarity on normalized unigrams
    visualize_similarity_matrix(similarity_unigram_norm_df, output_path,
                                title="Cosine Similarity between Programmers (Normalized Unigrams)")

    # Optional: Clustering Programmers Based on Unigram Frequencies
    clusters_unigram = cluster_programmers(unigram_sparse, programmers, output_path, n_clusters=2)
    clusters_unigram.to_csv(os.path.join(output_path, 'similarity_metrics', 'programmer_clusters_unigrams.csv'))
    logger.info("Saved programmer clusters (unigrams) as CSV.")
    print("\nProgrammer Clusters (Unigrams):")
    print(clusters_unigram)

    # Visualize clustering results
    visualize_clustering(clusters_unigram, output_path, title="Programmer Clusters (Unigrams)")

    logger.info("Code analysis completed successfully.")


if __name__ == "__main__":
    main()