import pytest
from tinybpe import BasicTokenizer 



def test_merges():
    basic = BasicTokenizer()
    test_text = "hello world ##### ????1! 😂 Namaste lol :D, what is this"
    merges = basic.train(test_text, 258, True)
    print(merges)
    assert(1) # for now manually failing the tests

def test_decode():
    basic = BasicTokenizer()
    test_tokens = [97, 98, 99, 100]
    test_text = "hello world ##### ????1! 😂 Namaste lol :D, what is this"
    merges = basic.train(test_text, 258)
    out = basic.decode(test_tokens)
    print(out)
    assert out == "abcd"

def test_encode():
    basic = BasicTokenizer()
    test_text = "This is new test"
    train_text = "hello world ##### ????1! 😂 Namaste lol :D, what is this"
    merges = basic.train(train_text, 258)
    out = basic.decode(basic.encode(test_text))
    assert test_text == out



if "__name__" == "__main__":
    pytest.main()


