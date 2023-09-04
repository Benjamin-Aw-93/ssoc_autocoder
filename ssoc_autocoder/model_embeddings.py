from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate_embeddings(model, 
                        tokenizer, 
                        title,
                        text):
    """"
    Function to generate embeddings from the language model
    
    """

    # Check data type
    # if type(text) != str:
    #     raise TypeError("Please enter a string for the 'text' argument.")
    # if type(title) != str:
    #     raise TypeError("Please enter a string for the 'title' argument.")
    def embeddings_iterator(data_loader):
        try:
            output = []
            for batch in tqdm(data_loader):
                tokenized_text = tokenizer(batch,
                                        text_pair=None, 
                                        add_special_tokens=True, 
                                        max_length=512,
                                        padding='max_length',
                                        return_token_type_ids=True,
                                        truncation=True,
                                        return_tensors='pt'
                                    )
                del tokenized_text['token_type_ids']
                output.append(model(**tokenized_text).last_hidden_state[:, 0].detach())
            return torch.cat(output, dim=0)
        except:
            output=[]

            for batch in data_loader:
                encoded_input = tokenizer(batch, return_tensors='pt', )
                output = model(**encoded_input)

                output.append(model(**tokenized_text).last_hidden_state[:, 0].detach())

            return torch.cat(output, dim=0)
    title_loader = DataLoader(title,
                             batch_size=16,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=False)
    
    title_embeddings = embeddings_iterator(title_loader)
    del title_loader

    text_loader = DataLoader(text,
                             batch_size=16,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=False)
    text_embeddings = embeddings_iterator(text_loader)
    del text_loader

    return title_embeddings, text_embeddings

if __name__ == "__main__":
    # load artifacts, data, then generate and store embeddings
    # Sept 2023: loading models directly from huggingface. Will include the version for customised models
    # where we load our own re-trained models
    # example where we embed with a bert model:
    language_model_name = "bert-base-uncased"
    tokenizer_name = "bert-base-uncased"
    huggingface_model = {'tokenizer': tokenizer_name,
                        'model': language_model_name
                       }
    language_model = AutoModel.from_pretrained(huggingface_model['model'])
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model['tokenizer'])

    # load data
    train = pd.read_csv('data/inputs/train.csv')
    title = train['job_title']
    text = train['job_description']

    title_embeddings, text_embeddings = generate_embeddings(language_model, tokenizer, title, text)

    # save the embeddings
    # torch.save(title_embeddings, f'data/embeddings/embeddings_title_{lang_model.__class__.__name__}.pt')
    # torch.save(text_embeddings, f'data/embeddings/embeddings_text_{lang_model.__class__.__name__}.pt')