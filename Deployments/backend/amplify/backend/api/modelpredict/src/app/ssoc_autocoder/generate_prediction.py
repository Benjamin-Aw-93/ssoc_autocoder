# Importing libraries
import json
import torch
import numpy as np
from transformers import DistilBertTokenizer
import model_training

parameters = {
    'pretrained_tokeniser': 'D:\\Users\\benjamin\\my_enviro\\Models\\distilbert-tokenizer-pretrained',
    'model_parameters_path': 'D:\\Users\\benjamin\\my_enviro\\Models\\autocoder-4jan-pretrained3epoch-60epoch.pt',
    'device': 'cpu',
    'sequence_max_length': 512
}

model_parameters = {
    'pretrained_model': 'D:\\Users\\benjamin\\my_enviro\Models\\mcf-pretrained-3epoch',
    'model_on_local_disk': True,
    'max_level': 5
}


def import_ssoc_idx_encoding(filename):
    """
    Imports the SSOC-index encoding and formats it correctly.
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


def generate_prediction(model, tokenizer, text, target, encoding, parameters):
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

    predicted_1D = encoding['SSOC_1D']['idx_ssoc'][np.argmax(preds["SSOC_1D"].detach().numpy())]
    predicted_1D_proba = np.max(m(preds['SSOC_1D']).detach().numpy())
    predicted_2D = encoding['SSOC_2D']['idx_ssoc'][np.argmax(preds["SSOC_2D"].detach().numpy())]
    predicted_2D_proba = np.max(m(preds['SSOC_2D']).detach().numpy())
    predicted_3D = encoding['SSOC_3D']['idx_ssoc'][np.argmax(preds["SSOC_3D"].detach().numpy())]
    predicted_3D_proba = np.max(m(preds['SSOC_3D']).detach().numpy())
    predicted_4D = encoding['SSOC_4D']['idx_ssoc'][np.argmax(preds["SSOC_4D"].detach().numpy())]
    predicted_4D_proba = np.max(m(preds['SSOC_4D']).detach().numpy())
    predicted_5D = encoding['SSOC_5D']['idx_ssoc'][np.argmax(preds["SSOC_5D"].detach().numpy())]
    predicted_5D_proba = np.max(m(preds['SSOC_5D']).detach().numpy())

    print(f"Target: {target}")
    print(f"Model predicted 1D: {predicted_1D} ({predicted_1D_proba*100:.2f}%)")
    print(f"Model predicted 2D: {predicted_2D} ({predicted_2D_proba*100:.2f}%)")
    print(f"Model predicted 3D: {predicted_3D} ({predicted_3D_proba*100:.2f}%)")
    print(f"Model predicted 4D: {predicted_4D} ({predicted_4D_proba*100:.2f}%)")
    print(f"Model predicted 5D: {predicted_5D} ({predicted_5D_proba*100:.2f}%)")


weight = torch.load(parameters['model_parameters_path'],
                    map_location=torch.device(parameters['device']))

model = model_training.HierarchicalSSOCClassifier(model_parameters)

model.load_state_dict(weight)

tokenizer = DistilBertTokenizer.from_pretrained(parameters['pretrained_tokeniser'])

text = "Assist the Resident Engineer in carrying out supervision works on site and enforce site quality control while coordinating with client / main contractor / sub-contractor. Liaise with site representatives from other trades during executive of site works to ensure smooth and successful completion of structural works. Assist the Resident Engineer in keeping and maintaining site record on progress of works, instructions given to Contractors, materials delivered to site, daily activities and weather conditions at site, number of visitors at site, any other information as attached and required under the Malaysian Laws and Regulations. Review and study all documents and drawings issued for construction, and ensure coordination of details with other trades. Carry out other duties as directed by the Qualified Person or his representative."

test_target = 31124

encoding = import_ssoc_idx_encoding(
    "D:\\Users\\benjamin\\my_enviro\\Data\\Reference\\ssoc-idx-encoding.json")

generate_prediction(model, tokenizer, text, test_target, encoding, parameters)
