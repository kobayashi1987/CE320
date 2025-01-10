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

# Configure logging
logging.basicConfig(level=logging.INFO)
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

    def display_frequencies(self, freq_dict, title="N-gram Frequencies"):
        """
        Displays the frequencies of N-grams with a title.

        Parameters:
            freq_dict (dict or Counter): A dictionary mapping N-grams to frequencies.
            title (str): The title to display before the frequencies.
        """
        print(f"\n{title}:")
        for ng, freq in freq_dict.items():
            print(f"{ng}: {freq}")

    def plot_ngram_frequencies(self, freq_dict, title="N-gram Frequencies", top_n=10):
        """
        Plots the top N most common N-grams using matplotlib.

        Parameters:
            freq_dict (dict or Counter): A dictionary mapping N-grams to frequencies.
            title (str): The title of the plot.
            top_n (int): The number of top N-grams to display.
        """
        if not freq_dict:
            print("No N-grams to plot.")
            return

        # Get the top N most common N-grams
        if isinstance(freq_dict, Counter):
            most_common = freq_dict.most_common(top_n)
        else:
            most_common = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]

        if not most_common:
            print("No N-grams meet the criteria for plotting.")
            return

        # Separate N-grams and their counts
        ngrams, counts = zip(*most_common)

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.bar(ngrams, counts, color='skyblue')
        plt.title(title)
        plt.xlabel('N-grams')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

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

def visualize_similarity_matrix(similarity_df, title="Similarity Matrix", cmap='YlGnBu', fmt=".2f"):
    """
    Visualizes the similarity or distance matrix using a heatmap.

    Parameters:
        similarity_df (pd.DataFrame): DataFrame containing similarity or distance scores.
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
    plt.show()

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

def cluster_programmers(sparse_matrix, programmers, n_clusters=2):
    """
    Clusters programmers based on their N-gram frequency vectors using K-Means.

    Parameters:
        sparse_matrix (csr_matrix): Sparse matrix where rows represent N-grams and columns represent programmers.
        programmers (list): List of programmer identifiers.
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

def visualize_clustering(clusters, title="Programmer Clusters"):
    """
    Visualizes clustering results using a bar plot.

    Parameters:
        clusters (pd.Series): Series mapping programmers to clusters.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=clusters)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Programmers')
    plt.tight_layout()
    plt.show()

def create_frequency_matrices(unigram_freqs, bigram_freqs, programmers):
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

    print("\nUnigram Frequency Matrix:")
    print(unigram_matrix)

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

    print("\nBigram Frequency Matrix:")
    print(bigram_matrix)

    # Create mappings
    unigram_ngram_to_index, programmer_to_index_uni = create_mappings(unigram_matrix)
    bigram_ngram_to_index, programmer_to_index_bi = create_mappings(bigram_matrix)

    # Convert to sparse matrices
    unigram_sparse = convert_to_sparse_matrix(unigram_matrix)
    bigram_sparse = convert_to_sparse_matrix(bigram_matrix)

    # Optional: Save matrices to CSV for further analysis
    unigram_matrix.to_csv('unigram_frequency_matrix.csv')
    bigram_matrix.to_csv('bigram_frequency_matrix.csv')

    # Optional: Visualize the matrices
    visualize_frequency_matrix(unigram_matrix, title="Unigram Frequency Matrix", top_n=20)
    visualize_frequency_matrix(bigram_matrix, title="Bigram Frequency Matrix", top_n=20)

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

def visualize_frequency_matrix(freq_matrix, title="Frequency Matrix", top_n=20):
    """
    Visualizes the top N N-grams in the frequency matrix using a heatmap.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to visualize.
        title (str): The title of the plot.
        top_n (int): The number of top N-grams to display.
    """
    if freq_matrix.empty:
        print("Frequency matrix is empty. Nothing to plot.")
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
    plt.show()

def main():
    # Example code samples from different programmers
    code_samples = {
        'Alice': """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
""",
        'Bob': """
def add(x, y, z):
    return x + y + z

sum = add(10, 20, 30)
print(sum)
""",
        'Charlie': """
def calculate(num1, num2, num3):
    return num1 + num2 * num3

total = calculate(7, 8, 9)
print(total)
""",
        # Add more programmers and their code samples as needed
    }

    # Initialize dictionaries to hold N-gram frequencies
    unigram_freqs = {}
    bigram_freqs = {}

    # Process each programmer's code
    for programmer, code in code_samples.items():
        logger.info(f"Processing code for {programmer}...")
        generator = NGramGenerator(code)

        # Count unigrams
        uni_freq = generator.count_ngram_frequencies_counter(n=1)
        unigram_freqs[programmer] = uni_freq

        # Count bigrams
        bi_freq = generator.count_ngram_frequencies_counter(n=2)
        bigram_freqs[programmer] = bi_freq

    # Create frequency matrices and convert to sparse format
    frequency_data = create_frequency_matrices(unigram_freqs, bigram_freqs, list(code_samples.keys()))

    # Access the sparse matrices and mappings if needed
    unigram_sparse = frequency_data['unigram']['sparse_matrix']
    bigram_sparse = frequency_data['bigram']['sparse_matrix']
    unigram_mappings = frequency_data['unigram']['ngram_to_index']
    bigram_mappings = frequency_data['bigram']['ngram_to_index']

    # Example: Accessing a specific frequency
    # Print the frequency of the unigram 'add' for 'Alice'
    print("\nFrequency of unigram 'add' for Alice:", frequency_data['unigram']['matrix'].at['add', 'Alice'])

    # Compute Cosine Similarity between programmers based on unigrams
    similarity_unigram_df = compute_cosine_similarity(unigram_sparse, list(code_samples.keys()))
    print("\nCosine Similarity between Programmers (Unigrams):")
    print(similarity_unigram_df)

    # Visualize the similarity matrix for unigrams
    visualize_similarity_matrix(similarity_unigram_df, title="Cosine Similarity between Programmers (Unigrams)")

    # Compute Cosine Similarity between programmers based on bigrams
    similarity_bigram_df = compute_cosine_similarity(bigram_sparse, list(code_samples.keys()))
    print("\nCosine Similarity between Programmers (Bigrams):")
    print(similarity_bigram_df)

    # Visualize the similarity matrix for bigrams
    visualize_similarity_matrix(similarity_bigram_df, title="Cosine Similarity between Programmers (Bigrams)")

    # Compute Euclidean Distance between programmers based on unigrams
    euclidean_unigram_df = compute_euclidean_distance(unigram_sparse, list(code_samples.keys()))
    print("\nEuclidean Distance between Programmers (Unigrams):")
    print(euclidean_unigram_df)

    # Visualize the Euclidean distance matrix for unigrams
    visualize_similarity_matrix(euclidean_unigram_df, title="Euclidean Distance between Programmers (Unigrams)", cmap='Reds', fmt=".2f")

    # Compute Pearson Correlation between programmers based on unigrams
    pearson_unigram_df = compute_pearson_correlation(frequency_data['unigram']['matrix'], list(code_samples.keys()))
    print("\nPearson Correlation between Programmers (Unigrams):")
    print(pearson_unigram_df)

    # Visualize the Pearson correlation matrix for unigrams
    visualize_similarity_matrix(pearson_unigram_df, title="Pearson Correlation between Programmers (Unigrams)")

    # Compute Jaccard Similarity between programmers based on unigrams
    jaccard_unigram_df = compute_jaccard_similarity(frequency_data['unigram']['matrix'], list(code_samples.keys()))
    print("\nJaccard Similarity between Programmers (Unigrams):")
    print(jaccard_unigram_df)

    # Visualize the Jaccard similarity matrix for unigrams
    visualize_similarity_matrix(jaccard_unigram_df, title="Jaccard Similarity between Programmers (Unigrams)")

    # Optional: Normalize frequency matrices
    normalized_unigram = normalize_frequency_matrix(frequency_data['unigram']['matrix'])
    print("\nNormalized Unigram Frequency Matrix:")
    print(normalized_unigram)

    normalized_bigram = normalize_frequency_matrix(frequency_data['bigram']['matrix'])
    print("\nNormalized Bigram Frequency Matrix:")
    print(normalized_bigram)

    # Compute Cosine Similarity on Normalized Data
    similarity_unigram_norm_df = compute_cosine_similarity(convert_to_sparse_matrix(normalized_unigram), list(code_samples.keys()))
    print("\nCosine Similarity between Programmers (Normalized Unigrams):")
    print(similarity_unigram_norm_df)

    visualize_similarity_matrix(similarity_unigram_norm_df, title="Cosine Similarity between Programmers (Normalized Unigrams)")

    # Optional: Clustering Programmers Based on Unigram Frequencies
    clusters_unigram = cluster_programmers(unigram_sparse, list(code_samples.keys()), n_clusters=2)
    print("\nProgrammer Clusters (Unigrams):")
    print(clusters_unigram)

    # Visualize clustering results
    visualize_clustering(clusters_unigram, title="Programmer Clusters (Unigrams)")

if __name__ == "__main__":
    main()