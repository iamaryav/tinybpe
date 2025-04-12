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
    
    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """
        Given the bytes of text return the token ids
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # check if this pair present in merges 
            # get lowest index from that merges
            # So this merge will happen in order
            pair = min(stats, key= lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # Nothing to merge so breaking
            # Otherwise merge the lowest merge index
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    
    def encode_ordinary(self, text):
        """This encoding will ignore special tokens"""
        # split in the chunks of text based on the same pattern used in the training
        text_chunks = re.findall(self.pattern, text)
        # encode all chunks seperately and then add to the list
        ids = []
        for chunk in text_chunks:
            chunk_bytes = text_chunks.encode("utf-8") # converting it to raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids


    # TODO: Test this method
    def enocde(self, text, allowed_special="none_raise"):
        """
        This method wil handle the encoding the special token
        """

        # These if conditions if for how to handle special tokens present in input
        # if present then and if not present then
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} is not understood")
        
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + '|'.join(re.escape(k) for k in special) + ")"
        print(special_pattern)
        special_chunks = re.split(special_pattern, text)
        print(special_chunks)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
        
