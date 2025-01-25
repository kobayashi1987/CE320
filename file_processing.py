import os

import pandas as pd

from logging_setup import logger
from ngram_generator import NGramGenerator
from utils import create_mappings, convert_to_sparse_matrix


def process_input_files(input_dir: str):
    programmers = []
    unigram_freqs = {}
    bigram_freqs = {}

    for programmer_dir in os.listdir(input_dir):
        programmer_path = os.path.join(input_dir, programmer_dir)
        if os.path.isdir(programmer_path):
            programmers.append(programmer_dir)
            code_combined = read_code_files(programmer_path)

            # Generate N-grams
            generator = NGramGenerator(code_combined)
            unigram_freqs[programmer_dir] = generator.count_ngram_frequencies_counter(n=1)
            bigram_freqs[programmer_dir] = generator.count_ngram_frequencies_counter(n=2)

    return programmers, unigram_freqs, bigram_freqs


def read_code_files(directory: str) -> str:
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
