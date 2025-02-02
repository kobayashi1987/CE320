import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from datetime import datetime

# Configure seaborn aesthetics
sns.set(style="whitegrid")

# Define project root path
project_root = "/Users/jack/Desktop/project/pycharm/ce320"
input_path = os.path.join(project_root, "input")


def setup_logging(output_path):
    """
    Sets up logging to file and console.
    """
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'homogeneity_determination_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
    return logger


def read_pairwise_comparison(file_path):
    """
    Reads the pairwise comparison CSV into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read pairwise comparison data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error reading pairwise comparison data from {file_path}: {e}")
        return None


def calculate_homogeneity_metrics(df):
    """
    Calculates homogeneity metrics based on similarity scores.
    """
    metrics = {}

    # List of similarity columns to consider
    similarity_columns = [
        'Cosine_Similarity_Unigrams',
        'Cosine_Similarity_Bigrams',
        'Pearson_Correlation_Unigrams',
        'Pearson_Correlation_Bigrams',
        'Jaccard_Similarity_Unigrams',
        'Jaccard_Similarity_Bigrams'
    ]

    # Flatten all similarity scores into a single series for aggregate metrics
    all_similarities = df[similarity_columns].values.flatten()

    # Remove any potential NaN values
    all_similarities = all_similarities[~pd.isna(all_similarities)]

    # Calculate metrics
    metrics['Average_Similarity'] = all_similarities.mean()
    metrics['Median_Similarity'] = pd.Series(all_similarities).median()
    metrics['Variance_Similarity'] = all_similarities.var()
    metrics['Std_Deviation_Similarity'] = all_similarities.std()
    metrics['Min_Similarity'] = all_similarities.min()
    metrics['Max_Similarity'] = all_similarities.max()
    metrics['Range_Similarity'] = metrics['Max_Similarity'] - metrics['Min_Similarity']

    logging.info("Calculated Homogeneity Metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    return metrics, all_similarities


def save_homogeneity_metrics(metrics, output_path,input_path):
    """
    Saves the homogeneity metrics to a CSV file.
    """
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    output_file = os.path.join(output_path, 'similarity_metrics', 'homogeneity_metrics.csv')
    output_file_input = os.path.join(input_path, 'homogeneity_metrics.csv')
    metrics_df.to_csv(output_file, index=False)
    metrics_df.to_csv(output_file_input, index=False)
    logging.info(f"Saved homogeneity metrics to {output_file}")


def plot_similarity_distribution(similarities, output_path):
    """
    Plots the distribution of similarity scores.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'visualizations', 'similarity_score_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved similarity score distribution plot to {plot_path}")


def plot_similarity_boxplot(similarities, output_path):
    """
    Plots a boxplot of similarity scores.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=similarities, color='lightgreen')
    plt.title('Boxplot of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'visualizations', 'similarity_score_boxplot.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved similarity score boxplot to {plot_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Determine team homogeneity based on pairwise similarity comparisons.")
    parser.add_argument('-i', '--input', required=True, help="Path to the pairwise comparison CSV file.")
    parser.add_argument('-o', '--output', required=True, help="Path to the output directory for homogeneity results.")
    args = parser.parse_args()

    input_file = args.input
    output_path = args.output
    output_path_input = os.path.join(project_root, "input")

    # Setup logging
    logger = setup_logging(output_path)
    logger.info("Starting Homogeneity Determination Process...")

    # Validate input file
    if not os.path.isfile(input_file):
        logger.error(f"Input file '{input_file}' does not exist.")
        return

    # Read pairwise comparison data
    pairwise_df = read_pairwise_comparison(input_file)
    if pairwise_df is None:
        logger.error("Failed to read pairwise comparison data. Exiting.")
        return

    # Calculate homogeneity metrics
    metrics, all_similarities = calculate_homogeneity_metrics(pairwise_df)

    # Save homogeneity metrics
    save_homogeneity_metrics(metrics, output_path, input_path)

    # Generate visualizations
    plot_similarity_distribution(all_similarities, output_path)
    plot_similarity_boxplot(all_similarities, output_path)

    logger.info("Homogeneity Determination Process Completed Successfully.")


if __name__ == "__main__":
    main()