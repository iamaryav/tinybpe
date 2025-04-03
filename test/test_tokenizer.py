import pytest
from tinybpe import BasicTokenizer 



def test_merges():
    basic = BasicTokenizer()
    test_text = "hello world ##### ????1! 😂 Namaste lol :D, what is this"
    merges = basic.train(test_text, 258, True)
    print(merges)
    assert(0) # for now manually failing the tests


if "__name__" == "__main__":
    pytest.main()


