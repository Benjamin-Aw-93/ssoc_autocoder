# General libraries
import pandas as pd
from copy import deepcopy
import numpy as np
import time
from datetime import datetime
import json

# Neural network libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModel



def generate_encoding(reference_data, ssoc_colname='SSOC 2020'):
    """
    Generates encoding from SSOC 2020 to indices and vice versa.

    Creates a dictionary that maps 5D SSOCs (v2020) to indices
    as required by PyTorch for multi-class classification, as well
    as the opposite mapping for indices to SSOCs. This is to enable
    data to be used for training as well as for generation of new
    predictions based on unseen data.

    Training: SSOC_xD, ssoc_idx SSOC -> idx
    Prediction: SSOC_xD, idx_ssoc

    Where x is the level of interest.

    Args:
        reference_data: Pandas dataframe containing all SSOCs (v2020)
        ssoc_colname: Name of the SSOC column

    Returns:
        Dictionary containing the SSOC to index mapping (for preparing the
        dataset) and index to SSOC mapping (for interpreting the predictions),
        for each SSOC level from 1D to 5D.
    """

    # Initialise the dictionary object to store the encodings for each level
    encoding = {}

    # Iterate through each level from 1 to 5
    for level in range(1, 6):

        # Initialise a dictionary object to store the respective-way encodings
        ssoc_idx_mapping = {}

        # Slice the SSOC column by the level required, drop duplicates, and sort
        ssocs = list(np.sort(reference_data[ssoc_colname].astype(
            'str').str.slice(0, level).unique()))

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

def import_ssoc_idx_encoding(filename):
    """
    Imports the SSOC-index encoding and formats it correctly.
    
    Args:
        filename: Path to the SSOC-index encoding JSON file
    Returns:
        SSOC-index encoding correctly formatted as a dictionary
    """

    # Open and load the JSON file as a dictionary
    with open(filename) as json_file:
        unformatted_dict = json.load(json_file)
        
    # Initialise the output dictionary object
    output_dict = {}
    
    # Iterate through each SSOC level in the encoding
    for ssoc_level in unformatted_dict.keys():
        
        # Initialise the key:value pair for each SSOC level
        output_dict[ssoc_level] = {
            'ssoc_idx': {},
            'idx_ssoc': {}
        }
        
        # Directly copy of the ssoc_idx object since this is formatted correctly
        output_dict[ssoc_level]['ssoc_idx'] = unformatted_dict[ssoc_level]['ssoc_idx']
        
        # For the idx_ssoc object, format the key as a number instead of a string
        for key, value in unformatted_dict[ssoc_level]['idx_ssoc'].items():
            output_dict[ssoc_level]['idx_ssoc'][int(key)] = value
            
    return output_dict

def encode_dataset(data,
                   encoding,
                   colnames):
    """
    Encodes SSOCs into indices for the training data.

    Uses the generated encoding to encode the SSOCs into indices
    at each digit level for PyTorch training. Drops any other
    columns that are not relevant to the model training process.

    Args:
        data: Pandas dataframe of the training data with the correct SSOC
        encoding: Encoding for each SSOC level
        colnames: Dictionary of all column names

    Returns:
        Pandas dataframe with each digit SSOC encoded correctly
    """

    # Create a copy of the dataframe and filter out any nulls
    encoded_data = deepcopy(data)[data[colnames['SSOC']].notnull()]

    # Subset the dataframe for only the required 2 columns
    encoded_data = encoded_data[[colnames['job_title'], colnames['job_description'], colnames['SSOC']]]
    encoded_data.columns = ['Title', 'Text', 'SSOC']

    # For each digit, encode the SSOC correctly
    for ssoc_level, encodings in encoding.items():
        encoded_data[ssoc_level] = encoded_data['SSOC'].astype('str').str.slice(
            0, int(ssoc_level[5])).replace(encodings['ssoc_idx'])

    return encoded_data

# Create a new Python class to handle the additional complexity


class SSOC_Dataset(Dataset):
    """
    Class to represent dataset along with several attributes.

    ...

    Attributes
    ----------
    dataframe: pandas dataframe
        SSOC dataset of interest
    tokenizer: Transformer Tokenizer
        End to end tokenization
    max_len: int
        Max length of sequences, for padding
    colnames: str
        For Something

    Methods
    ----------
    __getitem__: dict
        Input Id, Input mask (for batching inputs) and xD representation
    __len__:
        Number of rows in dataframe
    """

    # Define the class attributes
    def __init__(self, dataframe, tokenizer, max_len, colnames, architecture):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.architecture = architecture

    # Define the iterable over the Dataset object
    def __getitem__(self, index):

        # Extract the job title and text
        # Note that these have been set when encoding the dataset
        title = self.data.Title[index]
        text = self.data.Text[index]

        # Pass in the data into the tokenizer
        title_inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        text_inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        # Extract the IDs and attention mask
        title_ids = title_inputs['input_ids']
        title_mask = title_inputs['attention_mask']
        text_ids = text_inputs['input_ids']
        text_mask = text_inputs['attention_mask']

        # Return all the outputs needed for training and evaluation
        if self.architecture == "hierarchical":
            return {
                'title_ids': torch.tensor(title_ids, dtype=torch.long),
                'title_mask': torch.tensor(title_mask, dtype=torch.long),
                'text_ids': torch.tensor(text_ids, dtype=torch.long),
                'text_mask': torch.tensor(text_mask, dtype=torch.long),
                'SSOC_1D': torch.tensor(self.data.SSOC_1D[index], dtype=torch.long),
                'SSOC_2D': torch.tensor(self.data.SSOC_2D[index], dtype=torch.long),
                'SSOC_3D': torch.tensor(self.data.SSOC_3D[index], dtype=torch.long),
                'SSOC_4D': torch.tensor(self.data.SSOC_4D[index], dtype=torch.long),
                'SSOC_5D': torch.tensor(self.data.SSOC_5D[index], dtype=torch.long),
            }
        elif self.architecture == "straight":
            return {
                'title_ids': torch.tensor(title_ids, dtype=torch.long),
                'title_mask': torch.tensor(title_mask, dtype=torch.long),
                'text_ids': torch.tensor(text_ids, dtype=torch.long),
                'text_mask': torch.tensor(text_mask, dtype=torch.long),
                'SSOC_5D': torch.tensor(self.data.SSOC_5D[index], dtype=torch.long),
            }

    # Define the length attribute

    def __len__(self):
        return self.len


def prepare_data(encoded_train,
                 encoded_test,
                 tokenizer,
                 colnames,
                 parameters):
    """
    Prepares the encoded data for training and validation.
    Dataloader allows dataset to be represented as a Python iterable in a map-style format

    Args:
        encoded_train: pandas dataframe
            Train data with SSOC encoded into indices
        encoded_test: pandas dataframe
            Test data with SSOC encoded into indices
        colnames: dic
            Column name mappings key: standardized, value: actual naming
        parameters: dic
            Captures base information such as hyperparameters, node workers numbers, ssoc_encoding

    Returns:
        training_loader: torch dataloader
            Training data in map style torch data format
        validation_loader: torch dataloader
            Validation data in map style torch data format


    """

    # Creating the dataset and dataloader for the neural network
    training_loader = DataLoader(SSOC_Dataset(encoded_train, tokenizer, parameters['sequence_max_length'], colnames, parameters['architecture']),
                                 batch_size=parameters['training_batch_size'],
                                 num_workers=parameters['num_workers'],
                                 shuffle=True,
                                 pin_memory=True)
    validation_loader = DataLoader(SSOC_Dataset(encoded_test, tokenizer, parameters['sequence_max_length'], colnames, parameters['architecture']),
                                   batch_size=parameters['training_batch_size'],
                                   num_workers=parameters['num_workers'],
                                   shuffle=True,
                                   pin_memory=True)

    return training_loader, validation_loader

class HierarchicalSSOCClassifier_V1(torch.nn.Module):
    """
    Class to represent NN architecture.
    Inherits torch.nn.Module, base class for all NN modules
    ...
    Attributes
    ----------
    training_parameters: pandas dataframe
        Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
    encoding: dicitonary
        Encoding for each SSOC level
    Methods
    ----------
    forward: dict
        predictions batch size by length of target SSOC_xD
    """

    def __init__(self, training_parameters, encoding):

        self.training_parameters = training_parameters
        self.encoding = encoding

        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super(HierarchicalSSOCClassifier_V1, self).__init__()

        from transformers.models.distilbert.configuration_distilbert import DistilBertConfig

        self.config = DistilBertConfig(
            id2label = encoding['SSOC_5D']['idx_ssoc'],
            label2id = encoding['SSOC_5D']['ssoc_idx']
        )

        self.base_model_prefix = 'l1'
        self.device = 'cpu'

        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_1D_count = len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Stack 1: Predicting 1D SSOC (9)
        if self.training_parameters['max_level'] >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )

        # Stack 2: Predicting 2D SSOC (42)
        if self.training_parameters['max_level'] >= 2:

            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 777
            n_dims_2d = 768 + SSOC_1D_count

            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, n_dims_2d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_2d, n_dims_2d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_2d, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_2D_count)
            )

        # Stack 3: Predicting 3D SSOC (144)
        if self.training_parameters['max_level'] >= 3:

            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 819
            n_dims_3d = 768 + SSOC_1D_count + SSOC_2D_count

            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_3D_count)
            )

        # Stack 4: Predicting 4D SSOC (413)
        if self.training_parameters['max_level'] >= 4:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 963
            n_dims_4d = 768 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count

            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_4D_count)
            )

        # Stack 5: Predicting 5D SSOC (997)
        if self.training_parameters['max_level'] >= 5:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_5d = 1376
            n_dims_5d = 768 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count

            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, SSOC_5D_count)
            )

    def get_input_embeddings(self):
        return self.l1.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask):

        # Obtain the sentence embeddings from the DistilBERT model
        embeddings = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = embeddings[0]
        X = hidden_state[:, 0]

        # Initialise a dictionary to hold all the predictions
        predictions = {}

        # 1D Prediction
        if self.training_parameters['max_level'] >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)

        # 2D Prediction
        if self.training_parameters['max_level'] >= 2:
            X = torch.cat((X, predictions['SSOC_1D']), dim=1)
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

        # 3D Prediction
        if self.training_parameters['max_level'] >= 3:
            X = torch.cat((X, predictions['SSOC_2D']), dim=1)
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)

        # 4D Prediction
        if self.training_parameters['max_level'] >= 4:
            X = torch.cat((X, predictions['SSOC_3D']), dim=1)
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)

        # 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            X = torch.cat((X, predictions['SSOC_4D']), dim=1)
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.training_parameters['max_level'] + 1)}

class HierarchicalSSOCClassifier_V2(torch.nn.Module):
    """
    Class to represent NN architecture.
    Inherits torch.nn.Module, base class for all NN modules
    ...

    Attributes
    ----------
    training_parameters: pandas dataframe
        Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
    encoding: dicitonary
        Encoding for each SSOC level

    Methods
    ----------
    forward: dict
        predictions batch size by length of target SSOC_xD

    """

    def __init__(self, training_parameters):

        self.training_parameters = training_parameters

        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super(HierarchicalSSOCClassifier_V2, self).__init__()

        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'], 
                                                  local_files_only = self.training_parameters['local_files_only'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_1D_count = 9 #len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = 42 #len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = 144 #len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = 413 #len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = 997 #len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Stack 1: Predicting 1D SSOC (9)
        if self.training_parameters['max_level'] >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(1536, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )

        # Stack 2: Predicting 2D SSOC (42)
        if self.training_parameters['max_level'] >= 2:

            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 1545
            n_dims_2d = 1536 + SSOC_1D_count

            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_2D_count)
            )

        # Stack 3: Predicting 3D SSOC (144)
        if self.training_parameters['max_level'] >= 3:

            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 1587
            n_dims_3d = 1536 + SSOC_1D_count + SSOC_2D_count

            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_3D_count)
            )

        # Stack 4: Predicting 4D SSOC (413)
        if self.training_parameters['max_level'] >= 4:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 1731
            n_dims_4d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count

            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_4D_count)
            )

        # Stack 5: Predicting 5D SSOC (997)
        if self.training_parameters['max_level'] >= 5:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_5d = 2144
            n_dims_5d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count

            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, 2048),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(2048, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1024, SSOC_5D_count)
            )

    def forward(self, title_ids, title_mask, text_ids, text_mask):

        # Obtain the sentence embeddings from the DistilBERT model
        # Do this for both the job title and description text
        title_embeddings = self.l1(input_ids = title_ids, attention_mask = title_mask)
        title_hidden_state = title_embeddings[0]
        title_vec = title_hidden_state[:, 0]

        text_embeddings = self.l1(input_ids = text_ids, attention_mask = text_mask)
        text_hidden_state = text_embeddings[0]
        text_vec = text_hidden_state[:, 0]

        # Concatenate both vectors together
        X = torch.cat((title_vec, text_vec), dim = 1)

        # Initialise a dictionary to hold all the predictions
        predictions = {}

        # 1D Prediction
        if self.training_parameters['max_level'] >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)

        # 2D Prediction
        if self.training_parameters['max_level'] >= 2:
            X = torch.cat((X, predictions['SSOC_1D']), dim=1)
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

        # 3D Prediction
        if self.training_parameters['max_level'] >= 3:
            X = torch.cat((X, predictions['SSOC_2D']), dim=1)
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)

        # 4D Prediction
        if self.training_parameters['max_level'] >= 4:
            X = torch.cat((X, predictions['SSOC_3D']), dim=1)
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)

        # 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            X = torch.cat((X, predictions['SSOC_4D']), dim=1)
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.training_parameters['max_level'] + 1)}

class HierarchicalSSOCClassifier_V2pt1(torch.nn.Module):
    """
    Class to represent NN architecture.
    Inherits torch.nn.Module, base class for all NN modules
    ...

    Attributes
    ----------
    training_parameters: pandas dataframe
        Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
    encoding: dicitonary
        Encoding for each SSOC level

    Methods
    ----------
    forward: dict
        predictions batch size by length of target SSOC_xD

    """

    def __init__(self, training_parameters):

        self.training_parameters = training_parameters

        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super(HierarchicalSSOCClassifier_V2pt1, self).__init__()

        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'], 
                                                  local_files_only = self.training_parameters['local_files_only'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_1D_count = 9 #len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = 42 #len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = 144 #len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = 413 #len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = 997 #len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Stack 1: Predicting 1D SSOC (9)
        if self.training_parameters['max_level'] >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(1536, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )

        # Stack 2: Predicting 2D SSOC (42)
        if self.training_parameters['max_level'] >= 2:

            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 1545
            n_dims_2d = 1536 + SSOC_1D_count

            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_2D_count)
            )

        # Stack 3: Predicting 3D SSOC (144)
        if self.training_parameters['max_level'] >= 3:

            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 1587
            n_dims_3d = 1536 + SSOC_1D_count + SSOC_2D_count

            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_3D_count)
            )

        # Stack 4: Predicting 4D SSOC (413)
        if self.training_parameters['max_level'] >= 4:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 1731
            n_dims_4d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count

            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_4D_count)
            )

        # Stack 5: Predicting 5D SSOC (997)
        if self.training_parameters['max_level'] >= 5:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_5d = 2144
            n_dims_5d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count

            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, 2048),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(2048, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1024, SSOC_5D_count)
            )

    def forward(self, title_ids, title_mask, text_ids, text_mask):

        # Obtain the sentence embeddings from the DistilBERT model
        # Do this for both the job title and description text
        title_embeddings = self.l1(input_ids = title_ids, attention_mask = title_mask)
        title_hidden_state = title_embeddings[0]
        title_vec = title_hidden_state[:, 0]

        text_embeddings = self.l1(input_ids = text_ids, attention_mask = text_mask)
        text_hidden_state = text_embeddings[0]
        text_vec = text_hidden_state[:, 0]

        # Concatenate both vectors together
        X = torch.cat((title_vec, text_vec), dim = 1)

        # Initialise a dictionary to hold all the predictions
        predictions = {}
        
        return self.training_parameters['max_level']

        # 1D Prediction
        if self.training_parameters['max_level'] >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)

        # 2D Prediction
        if self.training_parameters['max_level'] >= 2:
            X = torch.cat((X, predictions['SSOC_1D']), dim=1)
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

        # 3D Prediction
        if self.training_parameters['max_level'] >= 3:
            X = torch.cat((X, predictions['SSOC_2D']), dim=1)
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)

        # 4D Prediction
        if self.training_parameters['max_level'] >= 4:
            X = torch.cat((X, predictions['SSOC_3D']), dim=1)
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)

        # 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            X = torch.cat((X, predictions['SSOC_4D']), dim=1)
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.training_parameters['max_level'] + 1)}

class HierarchicalSSOCClassifier_V2pt2(torch.nn.Module):
    """
    Class to represent NN architecture.
    Inherits torch.nn.Module, base class for all NN modules
    ...

    Attributes
    ----------
    training_parameters: pandas dataframe
        Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
    encoding: dicitonary
        Encoding for each SSOC level

    Methods
    ----------
    forward: dict
        predictions batch size by length of target SSOC_xD

    """

    def __init__(self, training_parameters):

        self.training_parameters = training_parameters

        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super(HierarchicalSSOCClassifier_V2pt2, self).__init__()
       
        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'], 
                                                  local_files_only = self.training_parameters['local_files_only'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_1D_count = 9 #len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = 42 #len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = 144 #len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = 413 #len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = 997 #len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Stack 1: Predicting 1D SSOC (9)
        if self.training_parameters['max_level'] >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(1536, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )

        # Stack 2: Predicting 2D SSOC (42)
        if self.training_parameters['max_level'] >= 2:

            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 1545
            n_dims_2d = 1536 + SSOC_1D_count

            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_2D_count)
            )

        # Stack 3: Predicting 3D SSOC (144)
        if self.training_parameters['max_level'] >= 3:

            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 1587
            n_dims_3d = 1536 + SSOC_1D_count + SSOC_2D_count

            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_3D_count)
            )

        # Stack 4: Predicting 4D SSOC (413)
        if self.training_parameters['max_level'] >= 4:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 1731
            n_dims_4d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count

            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, SSOC_4D_count)
            )

        # Stack 5: Predicting 5D SSOC (997)
        if self.training_parameters['max_level'] >= 5:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_5d = 2144
            n_dims_5d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count

            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1024, SSOC_5D_count)
            )

    def forward(self, input_tensor):

        # Obtain the sentence embeddings from the DistilBERT model
        # Do this for both the job title and description text
#         title_embeddings = self.l1(input_ids = title_ids, attention_mask = title_mask)
#         title_hidden_state = title_embeddings[0]
#         title_vec = title_hidden_state[:, 0]

#         text_embeddings = self.l1(input_ids = text_ids, attention_mask = text_mask)
#         text_hidden_state = text_embeddings[0]
#         text_vec = text_hidden_state[:, 0]

        # Concatenate both vectors together
        # X = torch.cat((title_vec, text_vec), dim = 1)
        
        # return json.dumps(X.tolist())
        X = input_tensor
        
        # return X

        # Initialise a dictionary to hold all the predictions
        predictions = {}
                                        
                                    

        # 1D Prediction
        if self.training_parameters['max_level'] >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)
            


        # 2D Prediction
        if self.training_parameters['max_level'] >= 2:
            X = torch.cat((X, predictions['SSOC_1D']))
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

        # 3D Prediction
        if self.training_parameters['max_level'] >= 3:
            X = torch.cat((X, predictions['SSOC_2D']))
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)

        # 4D Prediction
        if self.training_parameters['max_level'] >= 4:
            X = torch.cat((X, predictions['SSOC_3D']))
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)

        # 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            X = torch.cat((X, predictions['SSOC_4D']))
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)
            

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.training_parameters['max_level'] + 1)}

class StraightThruSSOCClassifier(torch.nn.Module):
    """
    Class to represent 5D direct NN architecture.
    Inherits torch.nn.Module, base class for all NN modules
    ...

    Attributes
    ----------
    training_parameters: pandas dataframe
        Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
    encoding: dicitonary
        Encoding for each SSOC level

    Methods
    ----------
    forward: dict
        predictions batch size by length of target SSOC_xD

    """

    def __init__(self, training_parameters):

        self.training_parameters = training_parameters

        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super(StraightThruSSOCClassifier, self).__init__()

        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_5D_count = len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Remove all layers keep last stack
        self.ssoc_5d_stack = torch.nn.Sequential(
            torch.nn.Linear(1536, 1536),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1536, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 3072),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(3072, 3072),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(3072, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, SSOC_5D_count)
        )

    def forward(self, title_ids, title_mask, text_ids, text_mask):

        # Obtain the sentence embeddings from the DistilBERT model
        # Do this for both the job title and description text
        title_embeddings = self.l1(input_ids = title_ids, attention_mask=title_mask)
        title_hidden_state = title_embeddings[0]
        title_vec = title_hidden_state[:, 0]

        text_embeddings = self.l1(input_ids = text_ids, attention_mask=text_mask)
        text_hidden_state = text_embeddings[0]
        text_vec = text_hidden_state[:, 0]

        # Concatenate both vectors together
        X = torch.cat(title_vec, text_vec)

        # Initialise a dictionary to hold all the predictions
        predictions = {}

        # Direct 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(5, self.training_parameters['max_level'] + 1)}


def prepare_model(encoding, parameters):
    """
    Setting up NN architecture along with the additonal parameters such as loss functions and optimizers

    Args:
        encoding:
            Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
        parameters:
            Encoding for each SSOC level

    Returns:
        model: forward prop architecture
            NN architecture representation
        loss_function: torch object
            Cross entropy loss function multiclass loss calculation
        optimizer: torch object
            Optimization algorithm for stochastic gradient descent

    """

    # Setting NN architeture

    if parameters["architecture"] == "hierarchical":
        if parameters["version"] == "V1":
            model = HierarchicalSSOCClassifier_V1(parameters, encoding)
        elif parameters["version"] == "V2":
            model = HierarchicalSSOCClassifier_V2(parameters)
        elif parameters["version"] == "V2pt1":
            model = HierarchicalSSOCClassifier_V2pt1(parameters)
        elif parameters["version"] == 'V2pt2':
            model = HierarchicalSSOCClassifier_V2pt2(parameters)
    elif parameters["architecture"] == "straight":
        model = StraightThruSSOCClassifier(parameters)
    else:
        raise InputError(
            "Choose which model to run: Set in parameters hierarchical or straight through")

    model.to(parameters['device'])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=parameters['learning_rate'])
    optimizer.zero_grad(set_to_none=True)

    return model, loss_function, optimizer


def calculate_accuracy(big_idx, targets):
    """

    Args:
        big_idx:
        targets:

    Returns:

    """
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train_model(model,
                loss_function,
                optimizer,
                training_loader,
                validation_loader,
                parameters):
    """
    Iteration through each epoch, with the predefined batch size.

    Args: Outputs of prepare_model
        model:HierarchicalSSOCClassifier
            NN architecture
        loss_function: torch object
            Cross entropy loss function multiclass loss calculation
        optimizer: torch object
            Optimization algorithm for stochastic gradient descent
        training_loader: torch dataloader
            Training data in map style torch data format
        validation_loader: torch dataloader
            Validation data in map style torch data format
        parameters: dic
            Captures base information such as hyperparameters, node workers numbers, ssoc_encoding

    Returns: None

    """
    # Start the timer
    start_time = datetime.now()
    print(f"Training started on: {start_time.strftime('%d %b %Y - %H:%M:%S')}")
    print("====================================================================")

    # Iterate over each training epoch
    for epoch_num in range(1, parameters['epochs']+1):

        # Start the timer for the epoch
        epoch_start_time = datetime.now()
        print(f"> Epoch {epoch_num} started on: {epoch_start_time.strftime('%d %b %Y - %H:%M:%S')}")
        print("--------------------------------------------------------------------")

        tr_loss = 0
        tr_n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0

        # Set the NN to train mode
        model.train()

        # Start the timer for the batch
        batch_start_time = datetime.now()

        # Iterate over each batch
        for batch, data in enumerate(training_loader):

            # Extract the data
            title_ids = data['title_ids'].to(parameters['device'], dtype=torch.long)
            title_mask = data['title_mask'].to(parameters['device'], dtype=torch.long)
            text_ids = data['text_ids'].to(parameters['device'], dtype=torch.long)
            text_mask = data['text_mask'].to(parameters['device'], dtype=torch.long)

            # Run the forward prop
            predictions = model(title_ids, title_mask, text_ids, text_mask)

            # Iterate through each SSOC level
            for ssoc_level, preds in predictions.items():

                # Extract the correct target for the SSOC level
                targets = data[ssoc_level].to(parameters['device'], dtype=torch.long)

                # Compute the loss function using the predictions and the targets
                level_loss = loss_function(preds, targets)

                # Initialise the loss variable if this is the 1D level
                # Else add to the loss variable
                # Note the weights on each level
                if ssoc_level == 'SSOC_1D' and parameters["architecture"] == "hierarchical":
                    loss = level_loss * parameters['loss_weights'][ssoc_level]
                elif parameters["architecture"] == "hierarchical":
                    loss += level_loss * parameters['loss_weights'][ssoc_level]
                else:
                    # no need to do stratified penalisation, all errors are equally penalised
                    loss = level_loss

            # Use the deepest level predictions to calculate accuracy
            # Exploit the fact that the last preds object is the deepest level one
            top_probs, top_probs_idx = torch.max(preds.data, dim=1)
            tr_n_correct += calculate_accuracy(top_probs_idx, targets)

            # Add this batch's loss to the overall training loss
            tr_loss += loss.item()

            # Keep count for the batch steps and number of examples
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Run backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For every X number of steps, print some information
            batch_size_printing = 50
            if (batch + 1) % batch_size_printing == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (tr_n_correct * 100) / nb_tr_examples
                print(f">> Training Loss per {batch_size_printing} steps: {loss_step:.4f} ")
                print(f">> Training Accuracy per {batch_size_printing} steps: {accu_step:.2f}%")
                print(
                    f">> Batch of {batch_size_printing} took {(datetime.now() - batch_start_time).total_seconds() / 60:.2f} mins")
                batch_start_time = datetime.now()

        # Set the model to evaluation mode for the validation
        model.eval()

        va_n_correct = 0
        va_loss = 0
        nb_va_steps = 0
        nb_va_examples = 0

        # Disable the calculation of gradients
        with torch.no_grad():

            # Iterate over each batch
            for batch, data in enumerate(validation_loader):

                # Extract the data
                title_ids = data['title_ids'].to(parameters['device'], dtype=torch.long)
                title_mask = data['title_mask'].to(parameters['device'], dtype=torch.long)
                text_ids = data['text_ids'].to(parameters['device'], dtype=torch.long)
                text_mask = data['text_mask'].to(parameters['device'], dtype=torch.long)

                # Run the forward prop
                predictions = model(title_ids, title_mask, text_ids, text_mask)

                # Iterate through each SSOC level
                for ssoc_level, preds in predictions.items():

                    # Extract the correct target for the SSOC level
                    targets = data[ssoc_level].to(parameters['device'], dtype=torch.long)

                    # Compute the loss function using the predictions and the targets
                    level_loss = loss_function(preds, targets)

                    # Initialise the loss variable if this is the 1D level
                    # Else add to the loss variable
                    # Note the weights on each level
                    if ssoc_level == 'SSOC_1D':
                        loss = level_loss * parameters['loss_weights'][ssoc_level]
                    else:
                        loss += level_loss * parameters['loss_weights'][ssoc_level]

                # Use the deepest level predictions to calculate accuracy
                # Exploit the fact that the last preds object is the deepest level one
                top_probs, top_probs_idx = torch.max(preds.data, dim=1)
                va_n_correct += calculate_accuracy(top_probs_idx, targets)

                # Add this batch's loss to the overall training loss
                va_loss += loss.item()

                # Keep count for the batch steps and number of examples
                nb_va_steps += 1
                nb_va_examples += targets.size(0)

        print("--------------------------------------------------------------------")
        epoch_tr_loss = tr_loss / nb_tr_steps
        epoch_va_loss = va_loss / nb_va_steps
        epoch_tr_accu = (tr_n_correct * 100) / nb_tr_examples
        epoch_va_accu = (va_n_correct * 100) / nb_va_examples
        print(
            f"> Epoch {epoch_num} Loss = \tTraining: {epoch_tr_loss:.3f}  \tValidation: {epoch_va_loss:.3f}")
        print(
            f"> Epoch {epoch_num} Accuracy = \tTraining: {epoch_tr_accu:.2f}%  \tValidation: {epoch_va_accu:.2f}%")
        print(
            f"> Epoch {epoch_num} took {(datetime.now() - epoch_start_time).total_seconds() / 60:.2f} mins")
        print("====================================================================")

    end_time = datetime.now()
    print("Training ended on:", end_time.strftime("%d %b %Y - %H:%M:%S"))
    print(f"Total training time: {(end_time - start_time).total_seconds() / 3600:.2f} hours")

    return


def convert_others_SSOC(predicted_SSOC_with_proba, threshold, available_ssoc_codes):
    """
    Predicting on actual text, comparing with actual 5D SSOC value

    Args:
        predicted_SSOC_with_proba: zip
            listing of prediction and correspodning probabilities
        threshold: double
            threhold for X-D SSOC, if its below the threshold, covert occupation code to Others
        available_ssoc_codes: pd series
            All possible SSOC values

    Returns: zip of converted prediction and probabilities
    """
    out_prediction = []
    out_probability = []

    for prediction, probability in predicted_SSOC_with_proba:
        if probability < threshold and prediction[-1] != '9':
            new_prediction = prediction[:-1] + '9'
            if new_prediction in available_ssoc_codes.values:
                print(f'Converting {prediction} to {new_prediction}')
                out_prediction.append(new_prediction)
            else:
                out_prediction.append(prediction)
        else:
            out_prediction.append(prediction)
        out_probability.append(probability)

    return zip(out_prediction, out_probability)


def generate_prediction(model,
                        tokenizer,
                        text,
                        target,
                        parameters,
                        top_n_threshold,
                        available_ssoc_codes,
                        failsafe=True):
    """
    Predicting on actual text, comparing with actual 5D SSOC value

    Args:
        model:HierarchicalSSOCClassifier
            NN architecture
        tokenizer: Transformer Tokenizer
            End to end tokenization
        text: str
            Text to predict on
        target: str
            5D SSOC value
        parameters: dic
            Captures base information such as hyperparameters, node workers numbers, ssoc_encoding
        top_n_threshold: dic
            Settings for top n digits and thresholds for each X-D SSOC
        available_ssoc_codes:  pd series
            All possible SSOC values
        failsafe: Boolean
            collapsing SSOC under threshold to Others catagories


    Returns: dict of zipped prdictions and probabilities
    """
    tokenized = tokenizer(
        text=text,
        text_pair=None,
        add_special_tokens=True,
        max_length=parameters['sequence_max_length'],
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )

    test_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long)
    test_mask = torch.tensor([tokenized['attention_mask']], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        preds = model(test_ids, test_mask)
        m = torch.nn.Softmax(dim=1)

    predicted_1D_idx = preds["SSOC_1D"].detach().numpy().argsort()[
        0][::-1][:top_n_threshold["SSOC_1D"]["n_digit"]]
    predicted_1D = [encoding['SSOC_1D']['idx_ssoc'][idx] for idx in predicted_1D_idx]
    predicted_1D_proba_all = m(preds['SSOC_1D']).detach().numpy()[0]
    predicted_1D_proba = [predicted_1D_proba_all[idx] for idx in predicted_1D_idx]
    predicted_1D_with_proba = zip(predicted_1D, predicted_1D_proba)

    predicted_2D_idx = preds["SSOC_2D"].detach().numpy().argsort()[
        0][::-1][:top_n_threshold["SSOC_2D"]["n_digit"]]
    predicted_2D = [encoding['SSOC_2D']['idx_ssoc'][idx] for idx in predicted_2D_idx]
    predicted_2D_proba_all = m(preds['SSOC_2D']).detach().numpy()[0]
    predicted_2D_proba = [predicted_2D_proba_all[idx] for idx in predicted_2D_idx]
    predicted_2D_with_proba = zip(predicted_2D, predicted_2D_proba)

    predicted_3D_idx = preds["SSOC_3D"].detach().numpy().argsort()[
        0][::-1][:top_n_threshold["SSOC_3D"]["n_digit"]]
    predicted_3D = [encoding['SSOC_3D']['idx_ssoc'][idx] for idx in predicted_3D_idx]
    predicted_3D_proba_all = m(preds['SSOC_3D']).detach().numpy()[0]
    predicted_3D_proba = [predicted_3D_proba_all[idx] for idx in predicted_3D_idx]
    predicted_3D_with_proba = zip(predicted_3D, predicted_3D_proba)

    predicted_4D_idx = preds["SSOC_4D"].detach().numpy().argsort()[
        0][::-1][:top_n_threshold["SSOC_4D"]["n_digit"]]
    predicted_4D = [encoding['SSOC_4D']['idx_ssoc'][idx] for idx in predicted_4D_idx]
    predicted_4D_proba_all = m(preds['SSOC_4D']).detach().numpy()[0]
    predicted_4D_proba = [predicted_4D_proba_all[idx] for idx in predicted_4D_idx]
    predicted_4D_with_proba = zip(predicted_4D, predicted_4D_proba)

    predicted_5D_idx = preds["SSOC_5D"].detach().numpy().argsort()[
        0][::-1][:top_n_threshold["SSOC_5D"]["n_digit"]]
    predicted_5D = [encoding['SSOC_5D']['idx_ssoc'][idx] for idx in predicted_5D_idx]
    predicted_5D_proba_all = m(preds['SSOC_5D']).detach().numpy()[0]
    predicted_5D_proba = [predicted_5D_proba_all[idx] for idx in predicted_5D_idx]
    predicted_5D_with_proba = zip(predicted_5D, predicted_5D_proba)

    print(f"Target: {target}")
    print(f'Model top {top_n_threshold["SSOC_1D"]["n_digit"]} predicted 1D:')
    for predicted, prob in predicted_1D_with_proba:
        print(f'{predicted}: {prob*100:.2f}%')

    if failsafe:
        predicted_2D_with_proba = convert_others_SSOC(
            predicted_2D_with_proba, top_n_threshold["SSOC_2D"]["threshold"], available_ssoc_codes)

    print(f'Model top {top_n_threshold["SSOC_2D"]["n_digit"]} predicted 2D:')
    for predicted, prob in predicted_2D_with_proba:
        print(f'{predicted}: {prob*100:.2f}%')

    if failsafe:
        predicted_3D_with_proba = convert_others_SSOC(
            predicted_3D_with_proba, top_n_threshold["SSOC_3D"]["threshold"], available_ssoc_codes)

    print(f'Model top {top_n_threshold["SSOC_3D"]["n_digit"]} predicted 3D:')
    for predicted, prob in predicted_3D_with_proba:
        print(f'{predicted}: {prob*100:.2f}%')

    if failsafe:
        predicted_4D_with_proba = convert_others_SSOC(
            predicted_4D_with_proba, top_n_threshold["SSOC_4D"]["threshold"], available_ssoc_codes)

    print(f'Model top {top_n_threshold["SSOC_4D"]["n_digit"]} predicted 4D:')
    for predicted, prob in predicted_4D_with_proba:
        print(f'{predicted}: {prob*100:.2f}%')

    if failsafe:
        predicted_5D_with_proba = convert_others_SSOC(
            predicted_5D_with_proba, top_n_threshold["SSOC_5D"]["threshold"], available_ssoc_codes)

    print(f'Model top {top_n_threshold["SSOC_5D"]["n_digit"]} predicted 5D:')
    for predicted, prob in predicted_5D_with_proba:
        print(f'{predicted}: {prob*100:.2f}%')

    return {'SSOC_1D_prediction': predicted_1D_with_proba,
            'SSOC_2D_prediction': predicted_2D_with_proba,
            'SSOC_3D_prediction': predicted_3D_with_proba,
            'SSOC_4D_prediction': predicted_4D_with_proba,
            'SSOC_5D_prediction': predicted_5D_with_proba}
