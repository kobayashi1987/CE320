import os
import seaborn as sns
from matplotlib import pyplot as plt
from logging_setup import logger
from utils import validate_numeric_dataframe


def visualize_similarity_matrix(similarity_df, output_path, title="Similarity Matrix", cmap='YlGnBu', fmt=".2f"):
    """
    Visualizes the similarity or distance matrix using a heatmap.

    Parameters:
        similarity_df (pd.DataFrame): DataFrame containing similarity or distance scores.
        output_path (str): Path to save the heatmap image.
        title (str): Title of the heatmap.
        cmap (str): Colormap to use for the heatmap.
        fmt (str): String formatting code.
    """
    similarity_df = validate_numeric_dataframe(similarity_df, name="Similarity DataFrame")
    save_plot(sns.heatmap(similarity_df, annot=True, fmt=fmt, cmap=cmap, square=True), output_path, title)


def visualize_clustering(clusters, output_path, title="Programmer Clusters"):
    """
    Visualizes clustering results using a bar plot.

    Parameters:
        clusters (pd.Series): Series mapping programmers to clusters.
        output_path (str): Path to save the clustering visualization.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=clusters)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Programmers')
    plt.tight_layout()

    # Save the clustering plot
    clustering_filename = f"{title.replace(' ', '_').lower()}.png"
    clustering_path = os.path.join(output_path, 'visualizations', clustering_filename)
    plt.savefig(clustering_path)
    plt.close()
    logger.info(f"Saved clustering visualization: {clustering_filename}")


def visualize_frequency_matrix(freq_matrix, output_path, title="Frequency Matrix", top_n=20):
    """
    Visualizes the top N N-grams in the frequency matrix using a heatmap.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to visualize.
        output_path (str): Path to save the heatmap image.
        title (str): The title of the plot.
        top_n (int): The number of top N-grams to display.
    """
    if freq_matrix.empty:
        logger.warning("Frequency matrix is empty. Nothing to plot.")
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

    # Save the heatmap
    heatmap_filename = f"{title.replace(' ', '_').lower()}.png"
    heatmap_path = os.path.join(output_path, 'visualizations', heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved heatmap: {heatmap_filename}")


def save_plot(fig, output_path, title):
    filename = os.path.join(output_path, 'visualizations', f"{title.replace(' ', '_').lower()}.png")
    fig.figure.savefig(filename)
    plt.close(fig.figure)
    logger.info(f"Saved plot: {filename}")
