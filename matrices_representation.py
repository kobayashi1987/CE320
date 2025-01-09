import tokenize
import io
from token import tok_name
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix


class NGramGenerator:
    def __init__(self, code, desired_types=None):
        """
        Initializes the NGramGenerator with Python code.

        Parameters:
            code (str): The Python source code to analyze.
            desired_types (set, optional): A set of token types to keep.
                Defaults to {'NAME', 'NUMBER', 'STRING', 'OP'}.
        """
        self.code = code
        self.desired_types = desired_types if desired_types else {'NAME', 'NUMBER', 'STRING', 'OP'}
        self.tokens = self.tokenize_code()
        self.filtered_tokens = list(self.filter_tokens())  # Convert generator to list

    def tokenize_code(self):
        """
        Tokenizes the Python code using the tokenize module.

        Returns:
            generator: A generator of token tuples (token_type, token_string).
        """
        code_bytes = io.BytesIO(self.code.encode('utf-8'))
        try:
            for tok in tokenize.tokenize(code_bytes.readline):
                if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                    continue
                token_type = tok_name.get(tok.type, tok.type)
                token_string = tok.string
                yield (token_type, token_string)
        except tokenize.TokenError as e:
            print(f"Tokenization error: {e}")
            return

    def filter_tokens(self):
        """
        Filters tokens based on desired types.

        Returns:
            generator: A generator of filtered token tuples.
        """
        return (token for token in self.tokens if token[0] in self.desired_types)

    def generate_ngrams_generator(self, n=2):
        """
        Generates N-grams using a generator to handle large datasets efficiently.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Yields:
            tuple: An N-gram tuple.
        """
        if n <= 0:
            raise ValueError("N must be a positive integer.")
        token_strings = [token[1] for token in self.filtered_tokens]
        if n > len(token_strings):
            print(f"Warning: N={n} is greater than the number of tokens ({len(token_strings)}). No N-grams generated.")
            return
        for i in range(len(token_strings) - n + 1):
            ngram = tuple(token_strings[i:i + n])
            # Uncomment the next line for debugging N-gram generation
            # print(f"Generated {n}-gram: {ngram}")
            yield ngram

    def count_ngram_frequencies_counter(self, n=2):
        """
        Counts the frequency of each N-gram using collections.Counter.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        ngram_gen = self.generate_ngrams_generator(n)
        return Counter(ngram_gen)

    def display_frequencies(self, freq_dict, title="N-gram Frequencies"):
        """
        Displays the frequencies of N-grams with a title.

        Parameters:
            freq_dict (dict or Counter): A dictionary mapping N-grams to frequencies.
            title (str): The title to display before the frequencies.
        """
        print(f"\n{title}:")
        for ng, freq in freq_dict.items():
            print(f"{ng}: {freq}")

    def plot_ngram_frequencies(self, freq_dict, title="N-gram Frequencies", top_n=10):
        """
        Plots the top N most common N-grams using matplotlib.

        Parameters:
            freq_dict (dict or Counter): A dictionary mapping N-grams to frequencies.
            title (str): The title of the plot.
            top_n (int): The number of top N-grams to display.
        """
        if not freq_dict:
            print("No N-grams to plot.")
            return

        # Get the top N most common N-grams
        if isinstance(freq_dict, Counter):
            most_common = freq_dict.most_common(top_n)
        else:
            most_common = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]

        if not most_common:
            print("No N-grams meet the criteria for plotting.")
            return

        # Separate N-grams and their counts
        ngrams, counts = zip(*most_common)

        # Create labels by joining the tokens in each N-gram
        labels = [' '.join(ng) for ng in ngrams]

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.bar(labels, counts, color='skyblue')
        plt.title(title)
        plt.xlabel('N-grams')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


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
    return csr_matrix(freq_matrix.values)


def create_frequency_matrices(unigram_freqs, bigram_freqs, programmers):
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

    print("\nUnigram Frequency Matrix:")
    print(unigram_matrix)

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

    print("\nBigram Frequency Matrix:")
    print(bigram_matrix)

    # Create mappings
    unigram_ngram_to_index, programmer_to_index_uni = create_mappings(unigram_matrix)
    bigram_ngram_to_index, programmer_to_index_bi = create_mappings(bigram_matrix)

    # Convert to sparse matrices
    unigram_sparse = convert_to_sparse_matrix(unigram_matrix)
    bigram_sparse = convert_to_sparse_matrix(bigram_matrix)

    # Optional: Save matrices to CSV for further analysis
    unigram_matrix.to_csv('unigram_frequency_matrix.csv')
    bigram_matrix.to_csv('bigram_frequency_matrix.csv')

    # Optional: Visualize the matrices
    visualize_frequency_matrix(unigram_matrix, title="Unigram Frequency Matrix", top_n=20)
    visualize_frequency_matrix(bigram_matrix, title="Bigram Frequency Matrix", top_n=20)

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


def visualize_frequency_matrix(freq_matrix, title="Frequency Matrix", top_n=20):
    """
    Visualizes the top N N-grams in the frequency matrix using a heatmap.

    Parameters:
        freq_matrix (pd.DataFrame): The frequency matrix to visualize.
        title (str): The title of the plot.
        top_n (int): The number of top N-grams to display.
    """
    if freq_matrix.empty:
        print("Frequency matrix is empty. Nothing to plot.")
        return

    # Sum frequencies across programmers to find top N-grams
    top_ngrams = freq_matrix.sum(axis=1).sort_values(ascending=False).head(top_n).index
    top_freq_matrix = freq_matrix.loc[top_ngrams]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_freq_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Programmers')
    plt.ylabel('N-grams')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():
    # Example code samples from different programmers
    code_samples = {
        'Alice': """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
""",
        'Bob': """
def add(x, y):
    return x + y

sum = add(10, 20)
print(sum)
""",
        'Charlie': """
def add(num1, num2):
    return num1 + num2

total = add(7, 8)
print(total)
""",
        # Add more programmers and their code samples as needed
    }

    # Initialize dictionaries to hold N-gram frequencies
    unigram_freqs = {}
    bigram_freqs = {}

    # Process each programmer's code
    for programmer, code in code_samples.items():
        print(f"\nProcessing code for {programmer}...")
        generator = NGramGenerator(code)

        # Count unigrams
        uni_freq = generator.count_ngram_frequencies_counter(n=1)
        unigram_freqs[programmer] = uni_freq

        # Count bigrams
        bi_freq = generator.count_ngram_frequencies_counter(n=2)
        bigram_freqs[programmer] = bi_freq

    # Create frequency matrices and convert to sparse format
    frequency_data = create_frequency_matrices(unigram_freqs, bigram_freqs, code_samples.keys())

    # Access the sparse matrices and mappings if needed
    unigram_sparse = frequency_data['unigram']['sparse_matrix']
    bigram_sparse = frequency_data['bigram']['sparse_matrix']
    unigram_mappings = frequency_data['unigram']['ngram_to_index']
    bigram_mappings = frequency_data['bigram']['ngram_to_index']

    # Example: Accessing a specific frequency
    # Print the frequency of the unigram ('add',) for 'Alice'
    print("\nFrequency of unigram ('add',) for Alice:", frequency_data['unigram']['matrix'].at[('add',), 'Alice'])

    # Example: Compute Cosine Similarity between programmers based on unigrams
    from sklearn.metrics.pairwise import cosine_similarity

    # Convert sparse matrix to dense if it's small; otherwise, use sparse
    # Here, we'll use the sparse matrix directly
    similarity = cosine_similarity(unigram_sparse.transpose())
    similarity_df = pd.DataFrame(similarity, index=frequency_data['unigram']['matrix'].columns,
                                 columns=frequency_data['unigram']['matrix'].columns)

    print("\nCosine Similarity between Programmers (Unigrams):")
    print(similarity_df)

    # Visualize the similarity matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap='YlGnBu')
    plt.title("Cosine Similarity between Programmers (Unigrams)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()