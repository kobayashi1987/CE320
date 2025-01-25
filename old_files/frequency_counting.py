import tokenize
import io
from token import tok_name
from collections import Counter
from nltk import ngrams as nltk_ngrams



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
        self.filtered_tokens = self.filter_tokens()

    def tokenize_code(self):
        """
        Tokenizes the Python code using the tokenize module.

        Returns:
            list: A list of token tuples (token_type, token_string).
        """
        tokens = []
        code_bytes = io.BytesIO(self.code.encode('utf-8'))
        try:
            for tok in tokenize.tokenize(code_bytes.readline):
                if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                    continue
                token_type = tok_name.get(tok.type, tok.type)
                token_string = tok.string
                tokens.append((token_type, token_string))
        except tokenize.TokenError as e:
            print(f"Tokenization error: {e}")
        return tokens

    def filter_tokens(self):
        """
        Filters tokens based on desired types.

        Returns:
            list: A list of filtered token tuples.
        """
        return [token for token in self.tokens if token[0] in self.desired_types]

    def generate_ngrams_nltk(self, n=2):
        """
        Generates N-grams using NLTK's ngrams function.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            list: A list of N-gram tuples.
        """
        if n <= 0:
            raise ValueError("N must be a positive integer.")
        token_strings = [token[1] for token in self.filtered_tokens]
        if n > len(token_strings):
            print(f"Warning: N={n} is greater than the number of tokens ({len(token_strings)}). Returning empty list.")
            return []
        return list(nltk_ngrams(token_strings, n))

    def generate_ngrams_manual(self, n=2):
        """
        Generates N-grams manually without using external libraries.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            list: A list of N-gram tuples.
        """
        if n <= 0:
            raise ValueError("N must be a positive integer.")
        token_strings = [token[1] for token in self.filtered_tokens]
        if n > len(token_strings):
            print(f"Warning: N={n} is greater than the number of tokens ({len(token_strings)}). Returning empty list.")
            return []
        return [tuple(token_strings[i:i + n]) for i in range(len(token_strings) - n + 1)]

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
            yield tuple(token_strings[i:i + n])

    def count_ngram_frequencies_counter(self, ngrams):
        """
        Counts the frequency of each N-gram using collections.Counter.

        Parameters:
            ngrams (list or iterable): A list or generator of N-gram tuples.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        return Counter(ngrams)

    def count_ngram_frequencies_dict(self, ngrams):
        """
        Counts the frequency of each N-gram using a standard dictionary.

        Parameters:
            ngrams (list or iterable): A list or generator of N-gram tuples.

        Returns:
            dict: A dictionary mapping N-grams to their frequencies.
        """
        frequency_dict = {}
        for ng in ngrams:
            if ng in frequency_dict:
                frequency_dict[ng] += 1
            else:
                frequency_dict[ng] = 1
        return frequency_dict

    def display_ngrams(self, ngrams, title="N-grams"):
        """
        Displays the N-grams with a title.

        Parameters:
            ngrams (list or generator): The list or generator of N-gram tuples.
            title (str): The title to display before the N-grams.
        """
        print(f"\n{title}:")
        for ng in ngrams:
            print(ng)

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

    def get_ngrams(self, n=2, method='manual'):
        """
        Retrieves N-grams using the specified method.

        Parameters:
            n (int): The number of tokens in each N-gram.
            method (str): The method to use ('manual', 'nltk', or 'generator').

        Returns:
            list or generator: The generated N-grams.
        """
        if method == 'manual':
            return self.generate_ngrams_manual(n)
        elif method == 'nltk':
            return self.generate_ngrams_nltk(n)
        elif method == 'generator':
            return self.generate_ngrams_generator(n)
        else:
            raise ValueError("Invalid method. Choose from 'manual', 'nltk', or 'generator'.")


def main():
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
    python_code1 = "CE320 is a wonderful module. Riccardo is a wonderful teacher."

    # Initialize the NGramGenerator with the Python code
    generator = NGramGenerator(python_code)

    print("Filtered Tokens:")
    for token in generator.filtered_tokens:
        print(token)

    # Define N values for which to generate N-grams
    n_values = [1, 2, 3]

    for n in n_values:
        print(f"\n--- {n}-gram Analysis ---")

        # Generate N-grams using the manual method
        ngrams_manual = generator.generate_ngrams_manual(n)
        generator.display_ngrams(ngrams_manual, f"{n}-grams (Manual)")

        # Count frequencies using Counter
        freq_counter = generator.count_ngram_frequencies_counter(ngrams_manual)
        generator.display_frequencies(freq_counter, f"{n}-gram Frequencies (Counter)")

        # Count frequencies using standard dictionary
        freq_dict = generator.count_ngram_frequencies_dict(ngrams_manual)
        generator.display_frequencies(freq_dict, f"{n}-gram Frequencies (Dictionary)")

    # Example with generator method for larger N
    n = 2
    print(f"\n--- {n}-gram Analysis using Generator ---")
    ngrams_gen = generator.generate_ngrams_generator(n)
    generator.display_ngrams(ngrams_gen, f"{n}-grams (Generator)")

    # Since generators can only be iterated once, recreate it for frequency counting
    ngrams_gen = generator.generate_ngrams_generator(n)
    freq_counter_gen = generator.count_ngram_frequencies_counter(ngrams_gen)
    generator.display_frequencies(freq_counter_gen, f"{n}-gram Frequencies (Counter from Generator)")

if __name__ == "__main__":
    main()