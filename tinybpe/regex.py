"""
# TODO: Add Docs

"""





from .base import Tokenizer, get_stats, merge
import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        If pattern: str is not given will use by default GPT4 pattern
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text: str, vocab_size, log=False):
        assert vocab_size > 256, "Vocab size should be greater than 256"
        num_merges = vocab_size - 256 # Merges will be done as part of this training
        # spliting up the text in gpt format using gpt regex pattern
        text_chunks = re.findall(self.compiled_pattern, text) 
        # List of chunks, each chunk is a list that is seperated by regex pattern
        ids = [list(chunks.encode("utf-8")) for chunks in text_chunks] # Encoding text to utf-8

        # this will save all the merges
        merges = {}
        # creating vocab and mapping first 256 chars
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # count the number of time every consecutive pairs occurs
            stats = {}
            for chunks_ids in ids:
                # stats will get updated for all the chunks
                get_stats(chunks_ids, stats)

            # pair with the highest count
            pair = max(stats, key=stats.get)
            new_token = 256 + i # creating new token will be used in next merge
            # mergin the token in each chunk
            ids = [merge(chunk_ids, pair, new_token) for chunk_ids in ids]
            # save the merge
            merges[pair] =  new_token
            vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]

            # print the beautiful log
            if log:
                print(f"merge {i + 1}: {pair} -> {new_token} ({vocab[new_token]}) had {stats[pair]} occurrences")
        
        # save to the class
        self.merges = merges
        self.vocab = vocab
        # print(self.merges)
        # print(self.vocab)
    
    def register_special_tokens(self, special_tokens):
        # special token is dict of str -> int
        # e.g. {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
        

    def decode(self, ids):
        # given the list of ids(integers) return the python string
        part_bytes = []
        for id in ids:
            if id in self.vocab:
                part_bytes.append(self.vocab[id])
            elif id in self.register_special_tokens:
                part_bytes.append(self.register_special_tokens[id])
            else:
                raise ValueError(f"Invalid token id: {id}")
            text_bytes = b"".join(part_bytes)
            text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode_ordinary(self, text):
        """This encoding will ignore special tokens"""

    def enocde():
        pass