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


# --------------
# Base class for tokenizer

class Tokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
    
    def train(self, text, vocab_size, log=False):
        # Tokenizer will train a vocab of size vocab_size form text
        raise NotImplementedError
