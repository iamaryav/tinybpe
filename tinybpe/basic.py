"""
Very Basic Tokenizer means Tiny
Algorithm matches with GPT tokenizer

Regex pattern to split token into fixed structure is not implemented

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
            tokens = merge(tokens, top_pair, new_token)
            merges[(top_pair)] = new_token
            # vocab list to save new tokens maps to pair of string Ex - {(256): ab
            vocab[new_token] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if log:
                print(f"{i + 1} merging pair: {top_pair} into new token: {new_token} ({vocab[new_token]}) with {stats[top_pair]} occurrences")

        self.merges = merges # Used in encode()
        self.vocab = vocab   # Used in decode()
    
    # Implement below methods
    def decode(self, tokens):
        """
        Given list of tokens and returns the string
        Example: (97, 97) -> "aa"
        """
        text = b"".join([self.vocab[idx] for idx in tokens])
        text = text.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list[int]:
        """
        Given the text return the list of tokens ids
        """

        text_bytes = text.encode("utf-8")
        # converting bytes to int between 0...256
        ids = list(text_bytes)
        # Loop will run until we have only 2 tokens
        # or the pair is not found in merges
        while len(ids) >= 2:
            stats = get_stats(ids)
            # from stats get the pair that was merge in the first in training the tokenizer
            # in self.merges
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # if pair is not found then break, because we have nothing to merge
            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids