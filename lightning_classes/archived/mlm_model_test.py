from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import random
import argparse


class ModelTest():
    """
    To load test data, model to be tested and generate summary of the model prediction.

    Returns:
        A class for model testing
    """

    def __init__(self, dataset):
        # load data
        self.data = dataset
        return None
    
    # tokenize dataset with tokenizer
    def tokenize(self, tokenizer):
        self.tokenized_data = self.data.map(lambda elem: tokenizer(elem['text']), remove_columns = ['text'])
        return None

    # perform masking of a single token for each row of data
    def single_mask(self, tokenizer):
        self.length = pd.Series([len(x['input_ids']) for x in self.tokenized_data['train']])
        self.masked_index = pd.Series(map(lambda num: random.randint(1, num-1), pd.Series(self.length)))
        self.masked_word = pd.Series(map(lambda index, encoded_text: tokenizer.decode(encoded_text['input_ids'][index]), self.masked_index, self.tokenized_data['train']))
        # replace single token with mask
        self.masked_data = pd.Series(map(lambda index, encoded_text: encoded_text['input_ids'][:index] + [tokenizer.mask_token_id] + encoded_text['input_ids'][index+1:], self.masked_index, self.tokenized_data['train']))
        # obtain a set of words surrounding the masked word for comparision later; offset by 4 tokens before and after masked word
        self.words_nearby_mask = map(lambda index, masked_text: tokenizer.decode(masked_text[index-4:index+4]), self.masked_index, self.masked_data)
        return None

    # generate the response of the model from the masked data
    def test(self, model, tokenizer):
        self.tokenize(tokenizer)
        self.single_mask(tokenizer)
        fill_mask = pipeline('fill-mask', model, tokenizer=tokenizer)
        self.unmasked_output = pd.Series(map(lambda masked_text: fill_mask(tokenizer.decode(masked_text)), self.masked_data))
        return None
    
    # get the top result and compare it with the masked token
    def summary(self):
        self.top_word = pd.Series(map(lambda result: result[0]['token_str'], self.unmasked_output))
        return pd.DataFrame({'masked_sentence': self.words_nearby_mask, 'masked_word': self.masked_word, 'top_guess': self.top_word})        


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type = str,
        help = 'path to model',
        default = "../model/"
    )

    parser.add_argument(
        '--testfile',
        type = str,
        help ='path to test file',
        default = "../testdata/pre-training data-sample.txt"
    )

    args = parser.parse_args()
    dataset = load_dataset('text', data_files=args.testfile)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    simpletest = ModelTest(dataset)
    simpletest.test(model, tokenizer)
    print(simpletest.summary())

    

if __name__ == "__main__":

    main()



