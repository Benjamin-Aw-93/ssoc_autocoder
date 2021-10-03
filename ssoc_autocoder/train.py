import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

import pandas as pd
import copy
import numpy as np

def generate_encoding(data,
                      ssoc_colname = 'SSOC 2020'):
    '''
    Generates encoding for SSOC to indices, as required by PyTorch
    for multi-class classification, for the training data

    Args:
        data: Pandas dataframe containing all SSOCs
        ssoc_colname: Name of the SSOC column

    Returns:
        Dictionary containing the SSOC to index mapping (for preparing the
        dataset) and index to SSOC mapping (for interpreting the predictions),
        for each SSOC level from 1D to 5D.
    '''

    # Initialise the dictionary object to store the encodings for each level
    encoding = {}

    # Iterate through each level from 1 to 5
    for level in range(1, 6):

        # Initialise a dictionary object to store the respective-way encodings
        ssoc_idx_mapping = {}

        # Slice the SSOC column by the level required, drop duplicates, and sort
        ssocs = list(np.sort(data[ssoc_colname].astype('str').str.slice(0, level).unique()))

        # Iterate through each unique SSOC (at i-digit level) and add to dict
        for i, ssoc in enumerate(ssocs):
            ssoc_idx_mapping[ssoc] = i

        # Add each level's encodings to the output dictionary
        encoding[f'SSOC_{level}D'] = {

            # Store the SSOC to index encoding
            'ssoc_idx': ssoc_idx_mapping,
            # Store the index to SSOC encoding
            'idx_ssoc': {v: k for k, v in ssoc_idx_mapping.items()}
        }

    return encoding

def encode_dataset(data,
                   encoding,
                   ssoc_colname = 'SSOC 2020'):
    '''
    Uses the generated encoding to encode the SSOCs at each
    digit level.

    Args:
        data: Pandas dataframe of the training data with the correct SSOC
        encoding: Encoding for each SSOC level
        ssoc_colname: Name of the SSOC column

    Returns:
        Pandas dataframe with each digit SSOC encoded correctly
    '''

    # Create a copy of the dataframe
    encoded_data = copy.deepcopy(data)

    # For each digit, encode the SSOC correctly
    for ssoc_level, encodings in encoding.items():
        encoded_data[ssoc_level] = encoded_data[ssoc_colname].astype('str').str.slice(0, int(ssoc_level[5])).replace(encodings['ssoc_idx'])

    return encoded_data

# Create a new Python class to handle the additional complexity
class SSOC_Dataset(Dataset):
    '''

    '''

    # Define the class attributes
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Define the iterable over the Dataset object
    def __getitem__(self, index):

        # Extract the text
        text = self.data[colnames['job_description']][index]

        # Pass in the data into the tokenizer
        inputs = self.tokenizer(
            text = text,
            text_pair = None,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True,
            return_token_type_ids = True,
            truncation = True
        )

        # Extract the IDs and attention mask
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Return all the outputs needed for training and evaluation
        return {
            'ids': torch.tensor(ids, dtype = torch.long),
            'mask': torch.tensor(mask, dtype = torch.long),
            'SSOC_1D': torch.tensor(self.data.SSOC_1D[index], dtype=torch.long),
            'SSOC_2D': torch.tensor(self.data.SSOC_2D[index], dtype=torch.long),
            'SSOC_3D': torch.tensor(self.data.SSOC_3D[index], dtype=torch.long),
            'SSOC_4D': torch.tensor(self.data.SSOC_4D[index], dtype=torch.long),
            'SSOC_5D': torch.tensor(self.data.SSOC_5D[index], dtype=torch.long),
        }

    # Define the length attribute
    def __len__(self):
        return self.len