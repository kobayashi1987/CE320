import tokenize
import io
from token import tok_name

# 1. Define a class to generate N-grams

class NGramGenerator:
    def __init__(self, code):
        """
        Initializes the NGramGenerator with Python code.

        Parameters:
            code (str): The Python source code to analyze.
        """
        self.tokens = self.tokenize_code(code)
        self.filtered_tokens = self.filter_tokens({'NAME', 'NUMBER', 'STRING', 'OP'})

# 2. Define a function to tokenize code
    def tokenize_code(self, code):
        tokens = []
        code_bytes = io.BytesIO(code.encode('utf-8'))
        try:
            for tok in tokenize.tokenize(code_bytes.readline):
                if tok.type == tokenize.ENCODING or tok.type == tokenize.ENDMARKER:
                    continue
                token_type = tok_name.get(tok.type, tok.type)
                token_string = tok.string
                tokens.append((token_type, token_string))
        except tokenize.TokenError as e:
            print(f"Tokenization error: {e}")
        return tokens

# 3. Define a function to filter tokens by type
    def filter_tokens(self, desired_types):
        """
        Filters tokens based on desired types.

        Parameters:
            desired_types (set): A set of token types to keep.

        Returns:
            list: A list of filtered token tuples.
        """
        return [token for token in self.tokens if token[0] in desired_types]

# 4. Define a function to generate n-grams using NLTK
    def generate_ngrams_nltk(self, n=2):
        """
        Generates N-grams using NLTK.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            list: A list of N-gram tuples.
        """
        from nltk import ngrams
        token_strings = [token[1] for token in self.filtered_tokens]
        return list(ngrams(token_strings, n))

# 5. Define a function to generate n-grams manually
    def generate_ngrams_manual(self, n=2):
        """
        Generates N-grams manually without external libraries.

        Parameters:
            n (int): The number of tokens in each N-gram.

        Returns:
            list: A list of N-gram tuples.
        """
        token_strings = [token[1] for token in self.filtered_tokens]
        ngrams_list = []
        for i in range(len(token_strings) - n + 1):
            ngram = tuple(token_strings[i:i + n])
            ngrams_list.append(ngram)
        return ngrams_list

# 6. Define a function to display N
    def display_ngrams(self, ngrams, title="N-grams"):
        """
        Displays the N-grams with a title.

        Parameters:
            ngrams (list): The list of N-gram tuples.
            title (str): The title to display before the N-grams.
        """
        print(f"\n{title}:")
        for ng in ngrams:
            print(ng)

# 7. Define a main function to test the NGramGenerator

def main():
    python_code = """ CE320 is a wonderful module. Riccardo is a wonderful teacher."""

    generator = NGramGenerator(python_code)

    print("Tokens:")
    for token in generator.filtered_tokens:
        print(token)

    # Generate Unigrams using manual method
    unigrams_manual = generator.generate_ngrams_manual(n=1)
    generator.display_ngrams(unigrams_manual, "Unigrams (Manual)")

    # Generate Bigrams using NLTK
    bigrams_nltk = generator.generate_ngrams_nltk(n=2)
    generator.display_ngrams(bigrams_nltk, "Bigrams (NLTK)")

    # Generate Trigrams using manual method
    trigrams_nltk = generator.generate_ngrams_nltk(n=3)
    generator.display_ngrams(trigrams_nltk, "Trigrams (NLTK)")


if __name__ == "__main__":
    main()