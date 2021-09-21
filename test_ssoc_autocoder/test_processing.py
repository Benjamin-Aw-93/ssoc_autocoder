import pytest
from ssoc_autocoder.processing import *


@pytest.fixture
def nlp():
    return spacy.load('en_core_web_lg')


def test_remove_prefix():
    assert remove_prefix("123 hello", ['123']) == "hello"
    assert remove_prefix("123 hello", ['@23']) == "123 hello"


def test_check_if_first_word_is_verb(nlp):
    assert check_if_first_word_is_verb('proven success', nlp) == False
    assert check_if_first_word_is_verb('review success', nlp) == True
    assert check_if_first_word_is_verb('acting on this', nlp) == True


if __name__ == '__main__':
    pytest.main()
