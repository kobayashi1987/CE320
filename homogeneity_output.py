import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from datetime import datetime
import networkx as nx

# Configure seaborn aesthetics
sns.set(style="whitegrid")


def setup_logging(output_path):
    """
    Sets up logging to file and console.
    """
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'homogeneity_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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


def read_csv_file(file_path, logger, description="CSV"):
    """
    Reads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read {description} from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading {description} from {file_path}: {e}")
        return None


def generate_pairwise_similarity_table(pairwise_df, output_path, logger):
    """
    Generates a formatted pairwise similarity table and saves it as CSV.
    """
    # Melt the DataFrame to have Programmer1, Programmer2, Metric, Value
    melted = pairwise_df.melt(id_vars=['Programmer1', 'Programmer2'], var_name='Metric', value_name='Value')

    # Pivot to have Metrics as separate columns
    pivot_df = melted.pivot_table(index=['Programmer1', 'Programmer2'], columns='Metric', values='Value').reset_index()

    # Save the pivot table
    output_file = os.path.join(output_path, 'similarity_metrics', 'pairwise_similarity_scores.csv')
    pivot_df.to_csv(output_file, index=False)
    logger.info(f"Saved pairwise similarity scores to {output_file}")

    # Optionally, display the table
    print("\nPairwise Similarity Scores:")
    print(pivot_df)


def generate_overall_homogeneity_score(homogeneity_metrics_df, output_path, logger):
    """
    Generates and saves the overall homogeneity score.
    """
    output_file = os.path.join(output_path, 'similarity_metrics', 'overall_homogeneity_score.csv')
    homogeneity_metrics_df.to_csv(output_file, index=False)
    logger.info(f"Saved overall homogeneity score to {output_file}")

    # Optionally, display the metrics
    print("\nOverall Homogeneity Metrics:")
    print(homogeneity_metrics_df)


def plot_similarity_network(pairwise_df, output_path, logger, threshold=0.5):
    """
    Plots a network graph of programmers where edges represent similarity above a threshold.
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes
    programmers = set(pairwise_df['Programmer1']).union(set(pairwise_df['Programmer2']))
    G.add_nodes_from(programmers)

    # Add edges with similarity scores above the threshold
    for _, row in pairwise_df.iterrows():
        prog1 = row['Programmer1']
        prog2 = row['Programmer2']
        similarity = row['Cosine_Similarity_Unigrams']  # Choose the metric to base the network on
        if similarity >= threshold:
            G.add_edge(prog1, prog2, weight=similarity)

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Draw edges
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw_networkx_edges(G, pos, width=[weight * 2 for weight in weights], alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Edge labels
    edge_labels = {(edge[0], edge[1]): f"{edge[2]['weight']:.2f}" for edge in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

    plt.title('Programmer Similarity Network')
    plt.axis('off')
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_path, 'visualizations', 'programmer_similarity_network.png')
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"Saved programmer similarity network graph to {plot_file}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate output reports for pairwise similarity and homogeneity.")
    parser.add_argument('-p', '--pairwise', required=True, help="Path to the pairwise comparison CSV file.")
    parser.add_argument('-m', '--metrics', required=True, help="Path to the homogeneity metrics CSV file.")
    parser.add_argument('-o', '--output', required=True, help="Path to the output directory for storing results.")
    args = parser.parse_args()

    pairwise_file = args.pairwise
    metrics_file = args.metrics
    output_path = args.output

    # Setup logging
    logger = setup_logging(output_path)
    logger.info("Starting Homogeneity Output Generation Process...")

    # Validate input files
    if not os.path.isfile(pairwise_file):
        logger.error(f"Pairwise comparison file '{pairwise_file}' does not exist.")
        return
    if not os.path.isfile(metrics_file):
        logger.error(f"Homogeneity metrics file '{metrics_file}' does not exist.")
        return

    # Read pairwise comparison data
    pairwise_df = read_csv_file(pairwise_file, logger, description="Pairwise Comparison Data")
    if pairwise_df is None:
        logger.error("Failed to read pairwise comparison data. Exiting.")
        return

    # Read homogeneity metrics data
    homogeneity_metrics_df = read_csv_file(metrics_file, logger, description="Homogeneity Metrics Data")
    if homogeneity_metrics_df is None:
        logger.error("Failed to read homogeneity metrics data. Exiting.")
        return

    # Generate Pairwise Similarity Table
    generate_pairwise_similarity_table(pairwise_df, output_path, logger)

    # Generate Overall Homogeneity Score
    generate_overall_homogeneity_score(homogeneity_metrics_df, output_path, logger)

    # Generate Similarity Network Graph
    plot_similarity_network(pairwise_df, output_path, logger, threshold=0.5)

    logger.info("Homogeneity Output Generation Process Completed Successfully.")


if __name__ == "__main__":
    main()