import pytest
from ssoc_autocoder.processing import *
import spacy
import json

# Loading the spacy object as a pytest fixture
@pytest.fixture
def nlp():
    return spacy.load('en_core_web_lg')

# On hold until we develop the full testing suite
# @pytest.fixture
# def example_strings():
#     with open('example_strings.txt') as f:
#         return json.load(f)

def test_remove_prefix():

    # Testing the prefix removal
    assert remove_prefix("123 hello", ['123']) == "hello"

    # Testing the prefix non-removal if the prefix is not found
    assert remove_prefix("123 hello", ['@23']) == "123 hello"

    # Testing the prefix removal for multiple prefixes
    assert remove_prefix("123 hello", ['test123', 'onetwothree', '123', '456']) == "hello"

    # Testing that we do not chain-remove multiple prefixes
    assert remove_prefix("123 hello world", ['test123', '123', 'hello']) == 'hello world'

def test_check_if_first_word_is_verb(nlp):

    # Testing the functionality
    assert check_if_first_word_is_verb('acting on this', nlp) == True

    # Testing the override False
    assert check_if_first_word_is_verb('proven success', nlp) == False

    # Testing the override True
    assert check_if_first_word_is_verb('review success', nlp) == True

def test_clean_raw_string():

    # Testing the removal of undesired characters
    assert clean_raw_string('hello \n this is a test') == 'hello  this is a test'
    assert clean_raw_string('hello \n&nbsp; this is&nbsp; a test&nbsp;') == 'hello  this is a test'
    assert clean_raw_string('\thello&rsquo;&rsquo; \n\t\xa0 this&amp; is\xa0 a test\t') == 'hello  this is a test'

    # Testing the replacement of known contractions
    assert clean_raw_string('hello \n this is No. 1 test') == 'hello  this is Number 1 test'

if __name__ == '__main__':
    pytest.main()
