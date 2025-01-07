
# 1. import necessary modules
import tokenize
import io
from token import tok_name
from nltk import ngrams

print ("ce320")
# 2. Define a function to tokenize code
def tokenize_code(code):
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
def filter_tokens(tokens, desired_types):
    return [token for token in tokens if token[0] in desired_types]

# 4. Define a function to generate n-grams
def generate_ngrams(tokens, n=2):
    token_strings = [token[1] for token in tokens]
    return list(ngrams(token_strings, n))




def main():

# Prepare the code to be tokenized
    python_code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
"""

    text1 = "CE320 is a wonderful module. Riccardo is a wonderful teacher."

    tokens = tokenize_code(text1)
    desired_types = {'NAME', 'NUMBER', 'STRING', 'OP'}
    filtered_tokens = filter_tokens(tokens, desired_types)

    print("Tokens:")
    for token in filtered_tokens:
        print(token)

    bigrams = generate_ngrams(filtered_tokens, n=2)
    print("\nBigrams:")
    for bg in bigrams:
        print(bg)


if __name__ == "__main__":
    main()