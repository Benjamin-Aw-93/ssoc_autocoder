import os
import json
from transformers import AutoTokenizer
import pickle
import torch
import ast
import json
from ssoc_autocoder.sagemaker_model_training import HierarchicalSSOCClassifier_V2pt2
from ssoc_autocoder.sagemaker_model_prediction import model_predict

def model_fn(model_dir):
    """
    Load the model for inference
    """

    pretrained_filepath_distilbert = os.path.join(model_dir, 'model/basemodel/')
    model_path = os.path.join(model_dir, 'model/ys_model_100.pth')
    tokenizer_path = os.path.join(model_dir, 'model/distilbert-tokenizer-pretrained-7epoch/')
    # # Load BERT tokenizer from disk.
    # with open(model_path+'ssoc-autocoder-model.pickle', 'rb') as handle:
    #     model = pickle.load(handle)
    param = {'pretrained_model':pretrained_filepath_distilbert, 
                      'max_level':5, 
                      'local_files_only':True, 
                      'sequence_max_length':512, 
                      'architecture':'hierarchical', 
                      'training_batch_size':32, 
                      'num_workers':4,
                      'version':'V2pt2',
                      'device':'cuda',
                      'learning_rate': 1e-3,
                      'epochs':1,
                      'loss_weights': {'SSOC_1D':20, 'SSOC_2D':10, 'SSOC_3D':4, 'SSOC_4D':2, 'SSOC_5D':1}}
    
        # Check if a GPU is available added this
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

              

    model = HierarchicalSSOCClassifier_V2pt2(param)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    

    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_dict = {'model': model, 'tokenizer':tokenizer}

    return model_dict

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    


        
    tokenizer = model['tokenizer']
    auto_model = model['model']
    
    
    # input_data = ast.literal_eval(input_data)
    # title = input_data['title']
    # text = input_data['text']
    tensor = input_data['input']
    
    encoding_path = 'code/data/ssoc-idx-encoding.json'
    

    preds = model_predict(auto_model, tokenizer, encoding_path, tensor)

    return preds

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """


    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response


    # if response_content_type == "application/json":
    #     # Convert the tensor to a list and then serialize to JSON
    #     prediction_json = json.dumps(prediction.tolist())
    #     return prediction_json
    # else:
    #     # For other content types, return the string representation of the tensor
    #     return str(prediction)