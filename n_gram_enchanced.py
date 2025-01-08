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
            desired_types (set, optional): A set of token types to keep. Defaults to {'NAME', 'NUMBER', 'STRING', 'OP'}.
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

    def count_ngram_frequencies(self, ngrams):
        """
        Counts the frequency of each N-gram in the list.

        Parameters:
            ngrams (list): A list of N-gram tuples.

        Returns:
            Counter: A Counter object mapping N-grams to their frequencies.
        """
        return Counter(ngrams)

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

    def get_ngrams(self, n=2, method='manual'):
        """
        Retrieves N-grams using the specified method.

        Parameters:
            n (int): The number of tokens in each N-gram.
            method (str): The method to use ('manual', 'nltk', 'generator').

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

# Example Usage
def main():
#     python_code = """
# def add(a, b):
#     return a + b
#
# result = add(5, 3)
# print(result)
#
# def subtract(a, b):
#     return a - b
#
# result = subtract(10, 4)
# print(result)
# """

    python_code = """ CE320 is a wonderful module. Riccardo is a wonderful teacher."""
    python_code1 = """
def complex_function(x, y, z):
    if x > y:
        return x * z
    elif y > z:
        return y * z
    else:
        return z * x

result = complex_function(10, 20, 30)
print(result)
"""

    generator = NGramGenerator(python_code1)

    print("Filtered Tokens:")
    for token in generator.filtered_tokens:
        print(token)

    # Define different N values to test flexibility
    n_values = [1, 2, 3, 4, 5]

    for n in n_values:
        print(f"\nGenerating {n}-grams using manual method:")
        ngrams_manual = generator.generate_ngrams_manual(n)
        generator.display_ngrams(ngrams_manual, f"{n}-grams (Manual)")

        print(f"\nGenerating {n}-grams using NLTK:")
        ngrams_nltk = generator.generate_ngrams_nltk(n)
        generator.display_ngrams(ngrams_nltk, f"{n}-grams (NLTK)")

        # Using generator method for higher N to demonstrate efficiency
        if n <= 3:  # Avoid overwhelming output for very large N
            print(f"\nGenerating {n}-grams using generator method:")
            ngrams_gen = generator.generate_ngrams_generator(n)
            generator.display_ngrams(ngrams_gen, f"{n}-grams (Generator)")

    # Example of counting N-gram frequencies
    print("\nCounting Bigram Frequencies:")
    bigrams = generator.generate_ngrams_manual(2)
    bigram_freq = generator.count_ngram_frequencies(bigrams)
    for bg, freq in bigram_freq.items():
        print(f"{bg}: {freq}")

if __name__ == "__main__":
        main()