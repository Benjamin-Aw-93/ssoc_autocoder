#!/usr/bin/env python
# coding=utf-8
#######################
# model_prediction.py
# Contains functions for generating predictions from a trained model
#######################

# Importing libraries
import json
import torch
from transformers import DistilBertTokenizer
from ssoc_autocoder import model_training

# Setting the default SSOC prediction parameters here
ssoc_prediction_parameters = {
    'SSOC_1D': {'top_n': 2, 'min_prob': 0.5},
    'SSOC_2D': {'top_n': 5, 'min_prob': 0.4},
    'SSOC_3D': {'top_n': 5, 'min_prob': 0.3},
    'SSOC_4D': {'top_n': 5, 'min_prob': 0.2},
    'SSOC_5D': {'top_n': 10, 'min_prob': 0.1}
}

def model_predict(pretrained_filepath,
                  model_filepath, 
                  tokenizer_filepath, 
                  ssoc_idx_encoding_filepath, 
                  text):

    """
    Main function that imports the model and generates prediction.

    Use this for deployment to AWS Lambda. Note that it doesn't include
    any options for passing in a target since it is assumed that we are
    generating predictions for unseen data.

    Args:
        pretrained_filepath: Path to the pretrained model (folder)
        model_filepath: Path to the model state_dict PyTorch object (.pt)
        tokenizer_filepath: Path to the pretrained DistilBERT tokenizer folder
        ssoc_idx_encoding_filepath: Path to the JSON file containing the SSOC encoding
        text: Text that needs a prediction
    Returns:
        Python dictionary of predictions and probabilities at each SSOC level
    """

    model_parameters = {
        'pretrained_model': pretrained_filepath,
        'local_files_only': True,
        'max_level': 5
    }

    # Initialise the model and tokenizer objects
    model = model_training.HierarchicalSSOCClassifier(model_parameters)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_filepath)

    # Reading in the SSOC-index encoding
    encoding = model_training.import_ssoc_idx_encoding(ssoc_idx_encoding_filepath)

    # Read in the trained parameters
    model.load_state_dict(torch.load(model_filepath))

    # Generate prediction
    prediction = generate_single_prediction(model, tokenizer, text, None, encoding)

    return prediction
def generate_embeddings(model, 
                        tokenizer, 
                        title,
                        text):
    # Check data type
    if type(text) != str:
        raise TypeError("Please enter a string for the 'text' argument.")
    if type(title) != str:
        raise TypeError("Please enter a string for the 'title' argument.")

    # Tokenize the title and text using the DistilBERT tokenizer
    tokenized_title = tokenizer(
        text = title,
        text_pair = None,
        add_special_tokens = True,
        max_length = 512,
        padding = 'max_length',
        return_token_type_ids = True,
        truncation = True
    )
    tokenized_text = tokenizer(
        text = text,
        text_pair = None,
        add_special_tokens = True,
        max_length = 512,
        padding = 'max_length',
        return_token_type_ids = True,
        truncation = True
    )
    
    # Extract the tensors from the tokenizer
    title_ids = torch.tensor([tokenized_title['input_ids']], dtype = torch.long)
    title_mask = torch.tensor([tokenized_title['attention_mask']], dtype = torch.long)
    text_ids = torch.tensor([tokenized_text['input_ids']], dtype = torch.long)
    text_mask = torch.tensor([tokenized_text['attention_mask']], dtype = torch.long)

    # Set the model to evaluation mode and generate the predictions
    model.eval()
    with torch.no_grad():
        title_vec = model.l1(title_ids, title_mask)[0][:, 0]
        text_vec = model.l1(text_ids, text_mask)[0][:, 0]
    
    embeddings_extracted = {
            'embeddings_title': title_vec.detach().numpy().tolist(),
            'embeddings_text': text_vec.detach().numpy().tolist()
        }
    
    return embeddings_extracted

def generate_single_prediction(model, 
                               tokenizer, 
                               text, 
                               target, 
                               encoding,
                               ssoc_prediction_parameters = ssoc_prediction_parameters):
        
    """
    Generates a single prediction from the trained neural network.
    
    -- to be filled in --
    """

    # Check data type
    if type(text) != str:
        raise TypeError("Please enter a string for the 'text' argument.")
    if type(target) != str:
        raise TypeError("Please enter a string for the 'target' argument.")

    # Tokenize the text using the DistilBERT tokenizer
    tokenized = tokenizer(
        text = text,
        text_pair = None,
        add_special_tokens = True,
        max_length = 512,
        padding = 'max_length',
        return_token_type_ids = True,
        truncation = True
    )
    
    # Extract the tensors from the tokenizer
    test_ids = torch.tensor([tokenized['input_ids']], dtype = torch.long)
    test_mask = torch.tensor([tokenized['attention_mask']], dtype = torch.long)
    
    # Set the model to evaluation mode and generate the predictions
    model.eval()
    with torch.no_grad():
        preds = model(test_ids, test_mask)
        m = torch.nn.Softmax(dim=1)
    
    # Iteratively generate predictions for each SSOC level that is specified
    predictions_with_proba = {}
    for ssoc_level, ssoc_level_params in sorted(ssoc_prediction_parameters.items()):
        
        # Extract the indices of the top n predicted SSOCs for the given SSOC level
        predicted_idx = preds[ssoc_level].detach().numpy().argsort()[0][::-1][:ssoc_level_params["top_n"]]
        
        # Extract the actual predicted probabilities from the softmax layer using the indices
        predicted_proba_all = m(preds[ssoc_level]).detach().numpy()[0]
        predicted_proba = [predicted_proba_all[idx] for idx in predicted_idx]
        
        # Convert the indices to the actual SSOC using the encoding dictionary
        predicted_ssoc = [encoding[ssoc_level]['idx_ssoc'][idx] for idx in predicted_idx]
        
        # If we already have the correct answer
        if target is not None:

            # Then check if the model made an accurate prediction
            # Meaning whether the correct SSOC appeared in the list of predictions
            accurate_prediction = False
            for ssoc in predicted_ssoc:
                if ssoc == target[0:len(ssoc)]:
                    accurate_prediction = True

        # If we are generating a prediction without a target
        else:

            # Then return a null for the accurate prediction value
            accurate_prediction = None
        
        # Append predictions with the predicted probability to the output
        predictions_with_proba[ssoc_level] = {
            'predicted_ssoc': predicted_ssoc,
            'predicted_proba': predicted_proba,
            'accurate_prediction': accurate_prediction
        }
        
    return predictions_with_proba

def generate_multiple_predictions(model, 
                                  tokenizer, 
                                  test_set,
                                  encoding,
                                  ssoc_prediction_parameters,
                                  ssoc_level = 'SSOC_5D'):
    
    """
    Wrapper around the generate_single_prediction function.

    -- to be filled in --
    """
        
    output = []
    accurate_predictions = []
    for i, row in test_set.iterrows():
        print(f'Generating prediction for {i+1}/{len(test_set)}...', end = '\r')
        predictions_with_proba = generate_single_prediction(model, 
                                                            tokenizer, 
                                                            row['description'], 
                                                            str(row['Predicted_SSOC_2020']),
                                                            encoding,
                                                            ssoc_prediction_parameters)
        output.append(predictions_with_proba)
        accurate_predictions.append(predictions_with_proba[ssoc_level]['accurate_prediction'])
    
    print('')
    accuracy = sum(accurate_predictions)/len(accurate_predictions)
    print(f'Overall {ssoc_level} accuracy: {accuracy:.2%}')
    
    return output
