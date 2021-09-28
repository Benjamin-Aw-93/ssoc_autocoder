import pytest
from ssoc_autocoder.processing import *
import spacy
import json

# Loading the spacy object as a pytest fixture


@pytest.fixture
def nlp():
    return spacy.load('en_core_web_lg')

# On hold until we develop the full testing suite


@pytest.fixture
def test_verb_check_txt():
    with open('test_ssoc_autocoder/test.txt') as f:
        text = json.load(f)
        text_out = text['test_verb_check']
        return text_out

@pytest.fixture
def integration_test_cases():
    with open('test_ssoc_autocoder/integration_test_cases.json') as f:
        integration_test_cases = json.load(f)
    return integration_test_cases

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
    assert clean_raw_string(
        '\thello&rsquo;&rsquo; \n\t\xa0 this&amp; is\xa0 a test\t') == 'hello  this is a test'

    # Testing the replacement of known contractions
    assert clean_raw_string('hello \n this is No. 1 test') == 'hello  this is Number 1 test'


def test_clean_html_unicode():

    # Test removal of normal tags
    assert clean_html_unicode('<p> test 1 </p>') == "test 1"

    # Test removal of tags with class
    assert clean_html_unicode('<p class = "class1"> test 2 </p>') == "test 2"

    # Test removal of numbered list
    assert clean_html_unicode('<p class = "class1"> 1. test 3 </p>') == "test 3"

    # Test removal of numbered list
    assert clean_html_unicode('1. test 3') == "test 3"

    # Test removal of symbols
    assert clean_html_unicode('test %4$ & *') == "test 4"

    # Test combination
    assert clean_html_unicode('<ol class = "new-list">2. test %5$ & ! </ol>') == "test 5"

def test_check_list_for_verbs(nlp, test_verb_check_txt):

    # Test ul/ol tags
    test_ans_case_0 = test_verb_check_txt[0]
    text1 = test_ans_case_0['input'][0]
    text2 = test_ans_case_0['input'][1]
    ans = test_ans_case_0['output']

    text_lst_0 = [BeautifulSoup(text1, 'html.parser').find(
        'ul'), BeautifulSoup(text2, 'html.parser').find('ul')]

    ans_0 = BeautifulSoup(ans, 'html.parser').find('ul')

    assert check_list_for_verbs(text_lst_0, nlp) == ans_0

    # Test p Tags
    test_ans_case_1 = test_verb_check_txt[1]
    text1 = test_ans_case_1['input'][0]
    ans = test_ans_case_1['output']

    text_lst_1 = [BeautifulSoup(text1, 'html.parser')]

    ans_1 = BeautifulSoup(ans, 'html.parser')

    assert check_list_for_verbs(text_lst_1, nlp) == ans_1

    # If verbs absent
    test_ans_case_2 = test_verb_check_txt[2]
    text1 = test_ans_case_2['input'][0]

    text_lst_2 = [BeautifulSoup(text1, 'html.parser').find('ul')]

    assert check_list_for_verbs(text_lst_2, nlp) == []

def test_process_text(integration_test_cases):
    
    for test_case in integration_test_cases:
        assert process_text(test_case['input']) == test_case['output']


if __name__ == '__main__':
    pytest.main()
