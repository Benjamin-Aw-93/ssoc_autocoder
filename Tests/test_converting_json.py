import pytest
from ssoc_autocoder.converting_json import *
import spacy
import json
import os
import pandas as pd
import pickle


# Loading json test answer json files as a pytest fixture
@pytest.fixture
def json_lst():

    lst = []

    for filename in os.listdir("json_test/Ans"):
        f = open("json_test/Ans/" + filename)
        entry = json.load(f)
        lst.append(entry)

    return lst


# Loading json test solutuion pickle files as a pytest fixture
@pytest.fixture
def json_ans():

    lst = []

    for filename in os.listdir("json_test/Sol"):
        f = open("json_test/Sol/" + filename, 'rb')
        entry = pickle.load(f)
        lst.append(entry)

    return lst


def test_extract_mcf_data(json_lst, json_ans):

    # Testing different variations, each with different missing information (1)
    assert extract_mcf_data(json_lst[0])[0] == json_ans[0]

    # Testing different variations, each with different missing information (2)
    assert extract_mcf_data(json_lst[1])[0] == json_ans[1]

    # Testing different variations, each with different missing information (3)
    assert extract_mcf_data(json_lst[3])[0] == json_ans[3]

    # Testing different variations, each with different missing information (4)
    assert extract_mcf_data(json_lst[4])[0] == json_ans[4]

    # Testing different variations, each with different missing information (5)
    assert extract_mcf_data(json_lst[5])[0] == json_ans[5]

    # Testing different variations, each with different missing information (6)
    assert extract_mcf_data(json_lst[6])[0] == json_ans[6]

    # Testing different variations, each with different missing information (7)
    assert extract_mcf_data(json_lst[2])[0] == json_ans[2]

    # Check if date is extracted corrected
    assert extract_mcf_data(json_lst[0])[1] == "2018-12-20"


if __name__ == '__main__':
    pytest.main()
