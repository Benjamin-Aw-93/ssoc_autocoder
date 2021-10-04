# General libraries
import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split

# Neural network libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

def generate_encoding(reference_data, ssoc_colname = 'SSOC 2020'):

    """
    Generates encoding from SSOC 2020 to indices and vice versa.

    Creates a dictionary that maps 5D SSOCs (v2020) to indices
    as required by PyTorch for multi-class classification, as well
    as the opposite mapping for indices to SSOCs. This is to enable
    data to be used for training as well as for generation of new
    predictions based on unseen data.

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
        ssocs = list(np.sort(reference_data[ssoc_colname].astype('str').str.slice(0, level).unique()))

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
    """
    Encodes SSOCs into indices for the training data.

    Uses the generated encoding to encode the SSOCs into indices
    at each digit level for PyTorch training.

    Args:
        data: Pandas dataframe of the training data with the correct SSOC
        encoding: Encoding for each SSOC level
        ssoc_colname: Name of the SSOC column

    Returns:
        Pandas dataframe with each digit SSOC encoded correctly
    """

    # Create a copy of the dataframe
    encoded_data = deepcopy(data)

    # For each digit, encode the SSOC correctly
    for ssoc_level, encodings in encoding.items():
        encoded_data[ssoc_level] = encoded_data[ssoc_colname].astype('str').str.slice(0, int(ssoc_level[5])).replace(encodings['ssoc_idx'])

    return encoded_data

# Create a new Python class to handle the additional complexity
class SSOC_Dataset(Dataset):
    """

    """

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
            padding = 'max_length',
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
            'SSOC_1D': torch.tensor(self.data.SSOC_1D[index], dtype = torch.long),
            'SSOC_2D': torch.tensor(self.data.SSOC_2D[index], dtype = torch.long),
            'SSOC_3D': torch.tensor(self.data.SSOC_3D[index], dtype = torch.long),
            'SSOC_4D': torch.tensor(self.data.SSOC_4D[index], dtype = torch.long),
            'SSOC_5D': torch.tensor(self.data.SSOC_5D[index], dtype = torch.long),
        }

    # Define the length attribute
    def __len__(self):
        return self.len

def prepare_data(encoded_data,
                 colnames,
                 parameters):
    """
    Prepares the encoded data for training and validation.



    Args:
        encoded_data:
        colnames: Dictionary of column names to
        parameters:

    Returns:

    """

    # Split the dataset into training and testing
    training_data, testing_data = train_test_split(encoded_data,
                                                   test_size = 0.2,
                                                   random_state = 2021)

    # Reset the indices as PyTorch refers to the indices when iterating
    training_data.reset_index(drop = True, inplace = True)
    testing_data.reset_index(drop = True, inplace = True)

    # Load the DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(parameters['pretrained_model'])

    # Creating the dataset and dataloader for the neural network
    training_loader = DataLoader(SSOC_Dataset(training_data, tokenizer, parameters['sequence_max_length']),
                                 batch_size = parameters['training_batch_size'],
                                 num_workers = parameters['num_workers'],
                                 shuffle = True)
    testing_loader = DataLoader(SSOC_Dataset(testing_data, tokenizer, parameters['sequence_max_length']),
                                batch_size = parameters['training_batch_size'],
                                num_workers = parameters['num_workers'],
                                shuffle = True)

    return training_loader, testing_loader

def prepare_model(encoding, parameters):
    """

    Args:
        encoding:
        parameters:

    Returns:

    """

    class HierarchicalSSOCClassifier(torch.nn.Module):

        def __init__(self):

            # Initialise the class, not sure exactly what this does
            super(HierarchicalSSOCClassifier, self).__init__()

            # Load the DistilBert model which will generate the embeddings
            self.l1 = DistilBertModel.from_pretrained(parameters['pretrained_model'])

            # Generate counts of each digit SSOCs
            SSOC_1D_count = len(encoding['SSOC_1D']['ssoc_idx'].keys())
            SSOC_2D_count = len(encoding['SSOC_2D']['ssoc_idx'].keys())
            SSOC_3D_count = len(encoding['SSOC_3D']['ssoc_idx'].keys())
            SSOC_4D_count = len(encoding['SSOC_4D']['ssoc_idx'].keys())
            SSOC_5D_count = len(encoding['SSOC_5D']['ssoc_idx'].keys())

            # Stack 1: Predicting 1D SSOC (9)
            if parameters['max_level'] >= 1:
                self.ssoc_1d_stack = torch.nn.Sequential(
                    torch.nn.Linear(768, 768),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(768, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, SSOC_1D_count)
                )

            # Stack 2: Predicting 2D SSOC (42)
            if parameters['max_level'] >= 2:

                # Adding the predictions from Stack 1 to the word embeddings
                n_dims_2d = 768 + SSOC_1D_count

                self.ssoc_2d_stack = torch.nn.Sequential(
                    torch.nn.Linear(n_dims_2d, n_dims_2d),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(n_dims_2d, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, SSOC_2D_count)
                )

        def forward(self, input_ids, attention_mask):

            # Obtain the sentence embeddings from the DistilBERT model
            embeddings = self.l1(input_ids = input_ids, attention_mask = attention_mask)
            hidden_state = embeddings[0]
            X = hidden_state[:, 0]

            # Initialise a dictionary to hold all the predictions
            predictions = {}

            # 1D Prediction
            if parameters['max_level'] >= 1:
                predictions['SSOC_1D'] = self.ssoc_1d_stack(X)

            # 2D Prediction
            if parameters['max_level'] >= 2:
                X = torch.cat((X, predictions['SSOC_1D']), dim = 1)
                predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

            return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, parameters['max_level'] + 1)}

    model = HierarchicalSSOCClassifier()
    model.to(parameters['device'])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = parameters['learning_rate'])

    return model, loss_function, optimizer


import time
from datetime import datetime


def calculate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train_model(model, loss_function, optimizer, epochs):
    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%d %b %Y - %H:%M:%S")
    print("Training started on:", current_time)

    for epoch in range(epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0

        epoch_start_time = time.time()
        batch_start_time = time.time()

        # Set the NN to train mode
        model.train()

        # Iterate over each batch
        for batch, data in enumerate(training_loader):

            # Extract the data
            ids = data['ids'].to(parameters['device'], dtype = torch.long)
            mask = data['mask'].to(parameters['device'], dtype = torch.long)

            # Run the forward prop
            predictions = model(ids, mask)

            # Iterate through each SSOC level
            for ssoc_level, preds in predictions.items():

                # Extract the correct target for the SSOC level
                targets = data[ssoc_level].to(parameters['device'], dtype = torch.long)

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
            top_probs, top_probs_idx = torch.max(preds.data, dim = 1)
            n_correct += calculate_accu(top_probs_idx, targets)

            # Calculate the loss
            #         targets_1d = data['targets_1d'].to(device, dtype = torch.long)
            #         targets_2d = data['targets_2d'].to(device, dtype = torch.long)
            #         loss1 = loss_function(preds_1d, targets_1d)
            #         loss2 = loss_function(preds_2d, targets_2d)
            #         loss = loss1*5 + loss2

            # Add this batch's loss to the overall training loss
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

            if (batch + 1) % 500 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Training Loss per 500 steps: {loss_step}")
                print(f"Training Accuracy per 500 steps: {accu_step}")
                print(f"Batch of 500 took {(time.time() - batch_start_time) / 60:.2f} mins")
                batch_start_time = time.time()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        print(f"Epoch training time: {(time.time() - epoch_start_time) / 60:.2f} mins")

    print(f"Total training time: {(time.time() - start_time) / 60:.2f} mins")
    now = datetime.now()
    current_time = now.strftime("%d %b %Y - %H:%M:%S")
    print("Training ended on:", current_time)

    return