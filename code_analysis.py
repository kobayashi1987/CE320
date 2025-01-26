import os

from logging_setup import setup_logging, logger
from metrics import compute_cosine_similarity, compute_euclidean_distance, compute_pearson_correlation, \
    compute_jaccard_similarity
from utils import validate_input_directory, create_output_subdirectories, process_programmer_data, \
    create_frequency_matrices, normalize_and_save_matrix, convert_to_sparse_matrix
from visualization import save_and_visualize_metric, cluster_and_visualize


def main():
    # efine project root path
    project_root = os.getcwd()

    # Define input and output paths
    input_path = os.path.join(project_root, "input")
    output_path = os.path.join(project_root, "output")

    # Declare output subdirectories
    OUTPUT_SUBDIRS = ['frequency_matrices', 'similarity_metrics', 'visualizations', 'logs']

    # Setup logging
    setup_logging(output_path)
    logger.info("Starting code analysis...")

    # Validate input directory
    if not validate_input_directory(input_path):
        return

    # Create necessary output subdirectories
    create_output_subdirectories(output_path, OUTPUT_SUBDIRS)

    # Initialise dictionaries
    programmers, unigram_freqs, bigram_freqs = process_programmer_data(input_path)

    if not programmers:
        logger.error("No valid programmer directories found in the input path.")
        return

    # Create frequency matrices and convert to sparse format
    frequency_data = create_frequency_matrices(unigram_freqs, bigram_freqs, programmers, output_path)

    for metric_func, name in [
        (compute_cosine_similarity, "Cosine Similarity"),
        (compute_euclidean_distance, "Euclidean Distance"),
        (compute_pearson_correlation, "Pearson Correlation"),
        (compute_jaccard_similarity, "Jaccard Similarity"),
    ]:
        save_and_visualize_metric(metric_func, frequency_data, name, programmers, output_path)

    normalized_unigram = normalize_and_save_matrix(frequency_data['unigram']['matrix'], 'normalized_unigram', output_path)

    normalized_sparse = convert_to_sparse_matrix(normalized_unigram)

    # Pass normalized_sparse to the metric function
    save_and_visualize_metric(
        compute_cosine_similarity,
        {'unigram': {'sparse_matrix': normalized_sparse}},
        "Cosine Similarity (Normalized Unigrams)",
        programmers,
        output_path,
    )

    unigram_clusters = cluster_and_visualize(frequency_data['unigram']['sparse_matrix'], programmers, output_path, n_clusters=2)
    unigram_clusters.to_csv(os.path.join(output_path, 'similarity_metrics', 'programmer_clusters_unigrams.csv'))

    logger.info("Code analysis completed successfully.")


if __name__ == "__main__":
    main()
