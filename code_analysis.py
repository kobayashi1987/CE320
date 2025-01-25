from file_processing import create_frequency_matrices
from ngram_generator import NGramGenerator
from metrics import (
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_pearson_correlation,
    compute_jaccard_similarity,
)
from clustering import cluster_programmers
from normalization import normalize_frequency_matrix
from utils import convert_to_sparse_matrix
from visualization import visualize_similarity_matrix, visualize_clustering
from logging_setup import setup_logging, logger

import os


def main():
    # Define project root path
    project_root = os.getcwd()

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
