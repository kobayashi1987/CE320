import os

import pandas as pd
from scipy.sparse import csr_matrix

from logging_setup import logger


def validate_output_path(output_path):
    """
    Ensures the output path exists; creates it if not.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)


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
    if not isinstance(freq_matrix, pd.DataFrame):
        raise ValueError("Expected a Pandas DataFrame for sparse matrix conversion.")
    return csr_matrix(freq_matrix.values)


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


