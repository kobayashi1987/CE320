import os

import pandas as pd
from scipy.sparse import csr_matrix

from clustering import cluster_programmers
from logging_setup import logger
from ngram_generator import NGramGenerator
from normalization import normalize_frequency_matrix


# Validation Utilities
def validate_input_directory(path):
    """
    Validates whether the given path is a directory.
    """
    if not os.path.isdir(path):
        logger.error(f"Input path '{path}' is not a valid directory.")
        return False
    return True


def create_output_subdirectories(base_path, subdirs):
    """
    Creates necessary subdirectories for output files.
    """
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


def validate_output_path(output_path):
    """
    Ensures the output path exists; creates it if not.
    """
    os.makedirs(output_path, exist_ok=True)


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


# File Processing Utilities
def process_programmer_data(input_path):
    """
    Reads and processes code files in the input directory for each programmer.

    Parameters:
        input_path (str): Path to the input directory.

    Returns:
        tuple: (list of programmers, unigram frequencies, bigram frequencies)
    """
    programmers, unigram_freqs, bigram_freqs = [], {}, {}
    for programmer_dir in os.listdir(input_path):
        programmer_path = os.path.join(input_path, programmer_dir)
        if os.path.isdir(programmer_path):
            programmers.append(programmer_dir)
            code_combined = read_code_files(programmer_path)
            generator = NGramGenerator(code_combined)
            unigram_freqs[programmer_dir] = generator.count_ngram_frequencies(n=1)
            bigram_freqs[programmer_dir] = generator.count_ngram_frequencies(n=2)
    return programmers, unigram_freqs, bigram_freqs


def read_code_files(directory):
    """
    Reads all code files from a directory and combines them into a single string.
    """
    code_combined = ""

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_combined += f.read() + "\n"
                except Exception as e:
                    logger.error(f"Failed to read file '{file_path}': {e}")
    return code_combined


# Frequency Matrix Utilities
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

    unigram_matrix = _generate_frequency_matrix(unigram_freqs, programmers, "Unigram")
    bigram_matrix = _generate_frequency_matrix(bigram_freqs, programmers, "Bigram")

    # Filter matrices to ensure columns match the programmers list
    unigram_matrix = unigram_matrix.loc[:, programmers]
    bigram_matrix = bigram_matrix.loc[:, programmers]

    # Convert to sparse matrices
    unigram_sparse = convert_to_sparse_matrix(unigram_matrix)
    bigram_sparse = convert_to_sparse_matrix(bigram_matrix)

    # Save frequency matrices as CSV
    frequency_dir = os.path.join(output_path, 'frequency_matrices')
    os.makedirs(frequency_dir, exist_ok=True)
    unigram_matrix.to_csv(os.path.join(frequency_dir, 'unigram_frequency_matrix.csv'))
    bigram_matrix.to_csv(os.path.join(frequency_dir, 'bigram_frequency_matrix.csv'))
    logger.info("Saved frequency matrices as CSV files.")

    return {
        'unigram': {
            'matrix': unigram_matrix,
            'sparse_matrix': unigram_sparse,
            'ngram_to_index': create_mappings(unigram_matrix)[0],
            'programmer_to_index': create_mappings(unigram_matrix)[1],
        },
        'bigram': {
            'matrix': bigram_matrix,
            'sparse_matrix': bigram_sparse,
            'ngram_to_index': create_mappings(bigram_matrix)[0],
            'programmer_to_index': create_mappings(bigram_matrix)[1],
        }
    }


def _generate_frequency_matrix(frequencies, programmers, name):
    """
    Generates a frequency matrix for N-grams.
    """
    all_ngrams = sorted(set(ngram for freq in frequencies.values() for ngram in freq.keys()))
    matrix = pd.DataFrame(0, index=all_ngrams, columns=programmers)

    for programmer, freq in frequencies.items():
        for ngram, count in freq.items():
            matrix.at[ngram, programmer] = count

    logger.info(f"{name} Frequency Matrix:")
    # logger.info(matrix)
    return matrix


# Conversion and Mapping Utilities
def create_mappings(freq_matrix):
    """
    Creates mappings from N-grams to indices and programmers to indices.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix with N-grams as rows and programmers as columns.

    Returns:
        tuple: (ngram_to_index, programmer_to_index)
    """
    return (
        {ngram: idx for idx, ngram in enumerate(freq_matrix.index)},
        {programmer: idx for idx, programmer in enumerate(freq_matrix.columns)},
    )


def convert_to_sparse_matrix(freq_matrix):
    """
    Converts a Pandas DataFrame to a SciPy CSR sparse matrix.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to convert.

    Returns:
        csr_matrix: The converted sparse matrix.
    """

    # Validation
    if not isinstance(freq_matrix, pd.DataFrame):
        raise ValueError("Expected a Pandas DataFrame for sparse matrix conversion.")

    return csr_matrix(freq_matrix.values)


def normalize_and_save_matrix(matrix, name, output_path):
    normalized_matrix = normalize_frequency_matrix(matrix)
    csv_path = os.path.join(output_path, 'frequency_matrices', f'{name}.csv')
    normalized_matrix.to_csv(csv_path)
    logger.info(f"Saved {name} as CSV.")
    return normalized_matrix


def filter_top_ngrams(freq_matrix, top_n=20):
    """
    Filters the frequency matrix to include only the top N most frequent N-grams (rows).

    Parameters:
        freq_matrix (pd.DataFrame): Original frequency matrix.
        top_n (int): Number of top N-grams to keep.

    Returns:
        pd.DataFrame: Filtered frequency matrix with top N-grams as rows.
    """
    if freq_matrix.empty:
        logger.warning("Frequency matrix is empty. Skipping filtering.")
        return freq_matrix

    # Sum the frequencies across all columns to determine the most frequent N-grams
    logger.info(f"Original Frequency Matrix shape: {freq_matrix.shape}")
    top_ngrams = freq_matrix.sum(axis=1).sort_values(ascending=False).head(top_n).index
    # Filter rows based on top N-grams
    filtered_matrix = freq_matrix.loc[top_ngrams, :]
    logger.info(f"Filtered Frequency Matrix shape: {filtered_matrix.shape}")
    return filtered_matrix
