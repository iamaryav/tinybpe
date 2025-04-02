# The goal is to make gpt-4 tokenizer based on what I learned


# Increase the vocab size
# 

# Counting how many times a pair appears in the text
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = 1 + counts.get(pair, 0)
    return counts 

# replacing most frequent pair with new token from vocab list
def merge(tokens, pair, new_token):
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

vocab_size = 50255 # Vocab size - matching the GPT-4 vocab size
num_merges = vocab_size - 256
ids = list(tokens) # making copy of input tokens so we don't change the actual token list
merges = {} # Save the merges that will help in decoding
for i in range(num_merges):
    stats = get_stats(ids)
    top_pair = max(stats, key=stats.get)
    new_token = 256 + i
    print(f"merging pair: {top_pair} into new token: {new_token}")
    ids = merge(ids, top_pair, new_token)
    merges[(top_pair)] = new_token