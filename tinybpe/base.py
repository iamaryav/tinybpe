import unicodedata

# Helper Methods 
def get_stats(ids):
    """
    Takes input list of Integer Example - [69, 420, 420, 42, 42, 42]
    Returns the count of pair Example - {(69, 420): 1, (420, 420): 1, (420, 42): 1, (42, 42): 2}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = 1 + counts.get(pair, 0)
    return counts 

# replacing most frequent pair with new token from vocab list
def merge(tokens, pair, new_token):
    """
    In the given list of tokens(Integers) and a pair replace that pair with new token
    returns a new token list after merge
    Example: [1, 1, 2, 3], (1, 1), 4 -> [4, 2, 3]

    """
    i = 0
    new_ids = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_ids.append(new_token)
            i += 2
        else:
            new_ids.append(tokens[i])
            i += 1
    return new_ids

def replace_control_chars(s: str) -> str:
    # This code is straight from Karpathy's minbpe repo
    # replace control character
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # escaping that
    return "".join(chars) 

def render_token(t: bytes) -> str:
    # Pretty print the output
    s = t.decode("utf-8", errors="replace")
    s = replace_control_chars(s)
    return s



# --------------
# Base class for tokenizer

class Tokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int e.g. {"<|endoftext|>": 100257}
        self.vocab = self._build_vocab() # int -> bytes
    
    def train(self, text, vocab_size, log=False):
        # Tokenizer will train a vocab of size vocab_size form text
        raise NotImplementedError
    
    def encode(self, text):
        # Tokenizer can encode string into list of Integer
        raise NotImplementedError
    
    def decode(self, ids):
        # Tokenizer can decode list of integer to string
        raise NotImplementedError

    def _build_vocab(self):
        # build the vocab list from merge list that will be used in decoding
        # from new token to bytes mapping
        vocab = {idx: bytes([idx]) for idx in range(256)} # creating the dict of first 0...255 integer and their corresponding bytes
        # all the new merges with their tokens
        for (p0, p1), id in self.merges.items():
            vocab[id] = vocab[p0] + vocab[p1]
        # All the specials token mapping
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    
    def save(self, file_prefix: str) -> None:
        # saves vocab and model in a file for later use

        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # tokenizer version
            f.write("tinybpe v1\n")
            # pattern
            f.write(f"{self.pattern}\n")
            # length of special token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens:
                f.write(f"{special} {idx}\n")
            # merges dict
            for pair, idx in self.merges:
                f.write(f"{pair} {idx}\n")
        # write vocab list to file
        vocab_file = file_prefix + ".vocab"
        # inverting the merges for new token lookup 
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w') as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find out this token is made which two token
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # If not the just write this file
                    f.write(f"[{s}] {idx}\n")
        pass

    
    def load(self, model_file):
        # Inverse of save but only for model file
        assert model_file.endswith(".model")
        # read the saved model file
        merges = {}
        special_token = {}
        idx = 256
        with open(model_file, 'r') as f:
            # read the version
            version = f.readline().strip()
            assert version == "tinybpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special token
            num_sepecial = int(f.readline().strip())
            for _ in range(num_sepecial):
                special, special_idx = f.readline().strip().split()
                special_token[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
            self.merges = merges
            self.special_tokens = special_token
            self.vocab = self._build_vocab()