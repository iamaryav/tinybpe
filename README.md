# tinybpe
 A simple byte pair encoder inspired by Karpathy's minbpe, built to understand and learn the concept better.  
 I wrote a blog to explain this in interesting ways checkout [here]()

### Installation from source
---
```
git clone https://github.com/iamaryav/tinybpe.git
cd tinybpe
```
After downloading this source code don't forget install the required libraries present in the requirements.txt

### Quick start
---
```python
from tinybpe import BasicTokenizer
basic = BasicTokenizer()
text = "This is new test"
basic.train(text, 255 + 4)
print(basic.encode(text))
# output:
# [258, 257, 257, 110, 101, 119, 32, 116, 101, 115, 116]

print(basic.decode([258, 257, 257, 110, 101, 119, 32, 116, 101, 115, 116]))
# output:
# This is new test

basic.save("test")
# saves the model in test.model file and vocab in test.vocab
```

### Training
---
By following below steps you can train your own tokenizer on any dataset with vocab size of 100k
```python
# Train basic tokenizer
from tinybpe import BasicTokenizer
train_text = "any text you want use"
text = "new test"
basic = BasicTokenizer()
basic.train(train_text, vocab_size=260)
basic.encode(text)# string -> tokens
basic.decode([2, 3, 4])# tokens -> String
basic.save("test")# creates test.model and test.vocab
basic.load("test.model") # loads the saved model 

# Train regex tokenizer
from tinybpe import BasicTokenizer
train_text = "any text you want use"
text = "new test"
regex_token = RegexTokenizer()
regex_token.train(train_text, vocab_size=260)
regex_token.encode(text)# string -> tokens
regex_token.decode([2, 3, 4])# tokens -> String
regex_token.save("test")# creates test.model and test.vocab
regex_token.load("test.model") # loads the saved model 

# Register special token using regex tokenizer
from tinybpe import RegexTokenizer 
train_text = "any text you want train on"
regex_token = RegexTokenizer()
regex_token.train(train_text, vocab_size=260)
# register any special token you want to add
regex_token.register_special_tokens({"<|endoftext|>": 260})
regex_token.encode("<|endoftext|>hello world", allowed_special="all")
```

### Inference: comparsion with GPT4 tokenizer
---
Comparsion between the tiktoken output and custom implementation of GPT4 tokenizer.  
Both produced same output.
```python
text = "This is new test"

import tiktoken
enc= tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# output
# [2028, 374, 502, 1296]

from tinybpe import GPT4Tokenizer
gpt = GPT4Tokenizer()
print(gpt.encode(text))
# output
# [2028, 374, 502, 1296]
```

### Tests
---
Pytest library has been used to test the tokenizer, install pytest lib using `pip install pytest` and execute below command to run all tests
```
pytest -v .
```

### Notes
---
- Tokenization.ipynb contains the step by step process to build this tokenizer from scratch

### Todos
---
- Working on future version of tokenizers and other existing tokenizer implementations.

### Wanna support? :)
----
- [X / Twitter](https://x.com/PriyaAryav)
- [Buy me a coffee](https://coff.ee/aryav)

### References
---
- [1] Wikipedia article [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding)
- [2] Karpathy's [minbpe](https://github.com/karpathy/minbpe)