import tokenize
import io
from token import tok_name
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from joblib import Parallel, delayed
import cProfile
import pstats


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
        self.tokens = list(self.tokenize_code())  # Convert to list for multiple iterations
        self.filtered_tokens = [token for token in self.tokens if token[0] in self.desired_types]

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
            yield from ()  # Yield nothing to ensure self.tokens is an empty list

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
        if len(self.filtered_tokens) < n:
            print(f"Warning: N={n} is greater than the number of tokens. No N-grams generated.")
            return  # Exit generator if not enough tokens

        token_strings = (token[1] for token in self.filtered_tokens)
        buffer = []
        for token in token_strings:
            buffer.append(token)
            if len(buffer) == n:
                yield tuple(buffer)
                buffer.pop(0)

    def count_ngram_frequencies_counter(self, n=2):
        """
        Counts the frequency of each N-gram using collections.Counter with generator.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        ngram_gen = self.generate_ngrams_generator(n)
        return Counter(ngram_gen)

    def count_ngram_frequencies_parallel(self, n=2):
        """
        Counts N-gram frequencies in parallel using multiprocessing.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        ngram_gen = self.generate_ngrams_generator(n)
        pool = Pool(cpu_count())

        # Define chunk size based on available memory and CPU cores
        chunk_size = 100000  # Adjust as needed

        def chunks(iterable, size):
            it = iter(iterable)
            while True:
                chunk = []
                try:
                    for _ in range(size):
                        chunk.append(next(it))
                except StopIteration:
                    if chunk:
                        yield chunk
                    break
                yield chunk

        partial_count = partial(Counter)
        results = pool.map(partial_count, chunks(ngram_gen, chunk_size))
        pool.close()
        pool.join()

        # Combine all partial Counters
        total_counter = Counter()
        for counter in results:
            total_counter.update(counter)
        return total_counter

    def count_ngram_frequencies_joblib(self, n=2, n_jobs=-1):
        """
        Counts N-gram frequencies in parallel using joblib.

        Parameters:
            n (int): The number of tokens in each N-gram.
            n_jobs (int): Number of parallel jobs. -1 uses all available cores.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        ngram_gen = self.generate_ngrams_generator(n)
        chunk_size = 100000  # Adjust as needed

        def chunks(iterable, size):
            it = iter(iterable)
            while True:
                chunk = []
                try:
                    for _ in range(size):
                        chunk.append(next(it))
                except StopIteration:
                    if chunk:
                        yield chunk
                    break
                yield chunk

        def count_ngrams_chunk(chunk):
            return Counter(chunk)

        chunks_list = list(chunks(ngram_gen, chunk_size))
        if not chunks_list:
            # No N-grams generated, return empty Counter
            return Counter()

        results = Parallel(n_jobs=n_jobs)(delayed(count_ngrams_chunk)(chunk) for chunk in chunks_list)
        total_counter = Counter()
        for counter in results:
            total_counter.update(counter)
        return total_counter

    def display_frequencies(self, freq_dict, title="N-gram Frequencies"):
        """
        Displays the frequencies of N-grams with a title.

        Parameters:
            freq_dict (dict or Counter): A dictionary mapping N-grams to frequencies.
            title (str): The title to display before the frequencies.
        """
        print(f"\n{title}:")
        if not freq_dict:
            print("No N-grams to display.")
            return
        for ng, freq in freq_dict.items():
            print(f"{ng}: {freq}")

    # Optional: You can implement a plotting method if needed
    # def plot_ngram_frequencies(self, freq_dict, title="N-gram Frequencies", top_n=5):
    #     import matplotlib.pyplot as plt
    #     most_common = freq_dict.most_common(top_n)
    #     ngrams, counts = zip(*most_common) if most_common else ([], [])
    #     plt.figure(figsize=(10, 5))
    #     plt.bar([' '.join(ng) for ng in ngrams], counts)
    #     plt.title(title)
    #     plt.xlabel("N-grams")
    #     plt.ylabel("Frequencies")
    #     plt.show()


def main():
    # Example Python code (can be replaced with a large codebase)
    python_code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)

def subtract(a, b):
    return a - b

result = subtract(10, 4)
print(result)
"""

    # Another example Python code snippet
    python_code1 = """CE320 is a wonderful module. Riccardo is a wonderful teacher."""

    # Natural language text (not valid Python code)
    text1 = """The two castaways, John Ferrier and the little girl who had shared his fortunes and had been adopted as his daughter, 
    accompanied the Mormons to the end of their great pilgrimage. 
    Little Lucy Ferrier was borne along pleasantly enough in Elder Stangerson’s waggon,
     a retreat which she shared with the Mormon’s three wives and with his son, a headstrong forward boy of twelve. Having rallied, 
     with the elasticity of childhood, she soon became a pet with the women, and reconciled herself to this new life in her moving canvas-covered home. In the meantime Ferrier having recovered from his privations, 
     distinguished himself as a useful guide and an indefatigable hunter. 
     So rapidly did he gain the esteem of his new companions, that when they reached the end of their wanderings, 
     it was unanimously agreed that he should be provided with as large and as fertile a tract of land as any of the settlers, with the exception of Young himself,
     and of Stangerson, Kemball, Johnston, and Drebber, who were the four principal Elders."""

    # Initialize the NGramGenerator with valid Python code
    generator = NGramGenerator(python_code)

    print("Filtered Tokens:")
    for token in generator.filtered_tokens:
        print(token)

    # Define N values for which to generate N-grams
    n_values = [1, 2, 3]

    # Define a frequency threshold (optional, currently not used)
    frequency_threshold = 1  # Change as needed

    for n in n_values:
        print(f"\n--- {n}-gram Analysis ---")

        # Frequency Counting using Counter
        freq_counter = generator.count_ngram_frequencies_counter(n)
        generator.display_frequencies(freq_counter, f"{n}-gram Frequencies (Counter)")

        # Frequency Counting using Parallel Processing
        freq_parallel = generator.count_ngram_frequencies_parallel(n)
        generator.display_frequencies(freq_parallel, f"{n}-gram Frequencies (Parallel Counter)")

        # Frequency Counting using Joblib
        freq_joblib = generator.count_ngram_frequencies_joblib(n)
        generator.display_frequencies(freq_joblib, f"{n}-gram Frequencies (Joblib Counter)")

        # Plotting Top 5 N-grams (Optional)
        # generator.plot_ngram_frequencies(freq_counter, title=f"Top 5 Most Common {n}-grams", top_n=5)

    # Profiling Example
    profiler = cProfile.Profile()
    profiler.enable()

    # Perform frequency counting
    freq = generator.count_ngram_frequencies_counter(2)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 functions by cumulative time


if __name__ == "__main__":
    main()