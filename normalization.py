import pandas as pd

from logging_setup import logger


def normalize_frequency_matrix(freq_matrix):
    """
    Normalizes the frequency matrix using Term Frequency (TF).
    Handles division by zero by replacing NaNs with zeros.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to normalize.

    Returns:
        pd.DataFrame: The normalized frequency matrix with no NaN values.
    """
    if not isinstance(freq_matrix, pd.DataFrame):
        raise ValueError("Expected a Pandas DataFrame for normalization.")

    # Normalize each column by its sum (Term Frequency normalization)
    normalized = freq_matrix.div(freq_matrix.sum(axis=0), axis=1)
    return normalized.fillna(0)  # Replace NaNs resulting from division by zero

