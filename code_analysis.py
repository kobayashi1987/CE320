import os

from logging_setup import setup_logging, logger
from metrics import compute_cosine_similarity, compute_euclidean_distance, compute_pearson_correlation, \
    compute_jaccard_similarity
from utils import validate_input_directory, create_output_subdirectories, process_programmer_data, \
    create_frequency_matrices
from visualization import save_and_visualize_metric, cluster_and_visualize, visualize_frequency_matrix


def main(input_dir=None, output_dir=None, metrics=None, clusters=None, ngram_value=2):
    """
    Main function for running the analysis.

    Parameters:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        metrics (list): List of metrics to compute (e.g., ["Cosine", "Euclidean"]).
        clusters (int): Number of clusters for clustering.
        ngram_value (int): Value of N for N-grams.
    """

    # Define project root path
    project_root = os.getcwd()

    # Define input and output paths
    input_path = os.path.join(project_root, "input")
    output_path = os.path.join(project_root, "output")

    # Validate input directory
    if not validate_input_directory(input_path):
        return

    # Setup logging
    setup_logging(output_path)
    logger.info(f"Starting code analysis with N={ngram_value}-grams...")

    # Create necessary output subdirectories
    OUTPUT_SUBDIRS = ['frequency_matrices', 'similarity_metrics', 'visualizations', 'logs']
    create_output_subdirectories(output_path, OUTPUT_SUBDIRS)

    # Process programmer data
    programmers, ngram_freqs = process_programmer_data(input_path, ngram_value)

    if not programmers:
        logger.error("No valid programmer directories found.")
        return

    # Create frequency matrix
    frequency_data = create_frequency_matrices(ngram_freqs, programmers, output_path, ngram_value)

    # Visualization and metric computation
    visualize_frequency_matrix(
        freq_matrix=frequency_data['matrix'],
        output_path=output_path,
        title=f"{ngram_value}-gram Frequency Matrix"
    )

    for metric_func, name in [
        (compute_cosine_similarity, "Cosine Similarity"),
        (compute_euclidean_distance, "Euclidean Distance"),
        (compute_pearson_correlation, "Pearson Correlation"),
        (compute_jaccard_similarity, "Jaccard Similarity"),
    ]:
        save_and_visualize_metric(metric_func, frequency_data, name, programmers, output_path, ngram_value)

    # Clustering Programmers
    clusters_df = cluster_and_visualize(frequency_data['sparse_matrix'], programmers, output_path, n_clusters=clusters)
    clusters_df.to_csv(os.path.join(output_path, 'similarity_metrics', 'programmer_clusters.csv'))

    logger.info("Code analysis completed successfully.")


if __name__ == "__main__":
    main()
