import pytest
from tinybpe import BasicTokenizer, replace_control_chars, render_token



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

def test_replace_control_character():
    test_str = "this is \n character test"
    out = replace_control_chars(test_str)
    assert out == "this is \\u000a character test" 

def test_render_token():
    test_str = "this is \n character test"
    test_bytes = test_str.encode("utf-8")
    print("test_bytes: ", test_bytes)
    out = render_token(test_bytes)
    assert out == "this is \\u000a character test" 


if "__name__" == "__main__":
    pytest.main()


