from .base import get_stats, merge



# Train the basice tokenizer
def train(tokens, vocab_size=256):
    assert(vocab_size >= 256, "vocab size should be greater than equal to 256")
    vocab_size = 256 # Vocab size - matching the GPT-4 vocab size
    num_merges = vocab_size - 256
    ids = list(tokens) # making copy of input tokens so we don't change the actual token list
    merges = {} # Save the merges that will help in decoding
    for i in range(num_merges):
        stats = get_stats(ids)
        top_pair = max(stats, key=stats.get)
        new_token = 256 + i
        # print(f"merging pair: {top_pair} into new token: {new_token}")
        ids = merge(ids, top_pair, new_token)
        merges[(top_pair)] = new_token