import io
import tokenize
from collections import Counter
from token import tok_name

from logging_setup import logger


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
                yield token_type, token_string
        except tokenize.TokenError as e:
            logger.error(f"Tokenization error: {e}")
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
            str: An N-gram string.
        """
        token_strings = [token[1] for token in self.filtered_tokens]
        if len(token_strings) < n:
            logger.warning(f"N={n} exceeds token count ({len(token_strings)}). No N-grams generated.")
            return
        yield from (' '.join(token_strings[i:i + n]) for i in range(len(token_strings) - n + 1))

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
