"""

TODO: Where is Docs bro?
Basic Tokenizer
Regex patter to split token into fixed structure is not implemented
Algorithm matches to GPT tokenizer

"""


from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    # Train the basice tokenizer
    def train(self, text, vocab_size, log=False):

        assert vocab_size > 256, "vocab size should be greater than 256"

        num_merges = vocab_size - 256 # Number of merges will be done as part of this training
        # converting text to utf-8 byte encoding
        text_bytes = text.encode("utf-8") # Encoding it to raw bytes
        tokens = list(text_bytes) # making to list of integers between 0 .... 255
        merges = {} # save the merges and token assignment: {(69, 420): 256}
        # int -> to bytes for first 256 
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # Get the current stats of token list after merges
            stats = get_stats(tokens)
            # find out the most frequent pair
            top_pair = max(stats, key=stats.get)
            new_token = 256 + i # New token used for current mrege
            ids = merge(tokens, top_pair, new_token)
            merges[(top_pair)] = new_token
            # vocab list to save new tokens maps to pair of string Ex - {(256): ab
            vocab[new_token] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if log:
                print(f"{i + 1} merging pair: {top_pair} into new token: {new_token} ({vocab[new_token]}) with {stats[top_pair]} occurrences")

        self.merges = merges # Used in encode()
        self.vocab = vocab   # Used in decode()
    
    # Implement below methods
    def decode(self, tokens):
        pass

    def encode(self, text):
        pass