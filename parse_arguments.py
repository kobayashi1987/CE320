import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="NAILER: Analyze and Compare Programming Styles")

    # Input directory
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the input folder containing programmer directories"
    )

    # Output directory
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Path to the output folder where results will be saved"
    )

    # Metrics to compute
    parser.add_argument(
        "-m", "--metrics",
        nargs="+",
        choices=["cosine", "euclidean", "pearson", "jaccard"],
        default=["cosine"],
        help="Similarity metrics to compute (default: cosine)"
    )

    # Clustering parameters
    parser.add_argument(
        "-c", "--clusters",
        type=int,
        default=2,
        help="Number of clusters for programmer grouping (default: 2)"
    )

    args = parser.parse_args()

    # Validate input and output paths
    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a valid directory.")
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    return args


def main():
    args = parse_arguments()

    # Example: Integration with the rest of your script
    print(f"Input Path: {args.input}")
    print(f"Output Path: {args.output}")
    print(f"Selected Metrics: {args.metrics}")
    print(f"Number of Clusters: {args.clusters}")

    # TODO: Integrate args into the analysis pipeline
    # For instance:
    # 1. Use args.input as the directory to read programmer data.
    # 2. Use args.output for saving results.
    # 3. Compute only the selected metrics specified in args.metrics.
    # 4. Perform clustering with args.clusters clusters.


if __name__ == "__main__":
    main()
