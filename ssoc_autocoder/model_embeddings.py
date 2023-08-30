from transformers import DistilBertTokenizer, AutoModel,DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import model_training
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
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
    title_loader = DataLoader(title,
                             batch_size=32,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=False)
    
    text_loader = DataLoader(text,
                             batch_size=32,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=False)
    
    def embeddings_iterator(data_loader):
        try:
            output = []
            for batch in data_loader:
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
                output.append(model(**tokenized_text).last_hidden_state[:, 0])
            return torch.cat(output, dim=0)
        except:
            output=[]

            for batch in data_loader:
                encoded_input = tokenizer(batch, return_tensors='pt', )
                output = model(**encoded_input)

                output.append(model(**tokenized_text).last_hidden_state[:, 0])

            return torch.cat(output, dim=0)
    
    title_embeddings = embeddings_iterator(title_loader)
    text_embeddings = embeddings_iterator(text_loader)
    
    return title_embeddings, text_embeddings

if __name__ == "__main__":
    # load artifacts, data, then generate and store embeddings
    # lang_model_path = 'artifacts/lang-model/'
    # tokenizer_path = 'artifacts/distilbert-tokenizer-pretrained-7epoch'
    # lang_model = AutoModel.from_pretrained(lang_model_path)
    # tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    language_models = {'DistilBERT':[DistilBertTokenizer.from_pretrained('./artifacts/distilbert-tokenizer-pretrained-7epoch'),DistilBertModel.from_pretrained('./artifacts/basemodel')],
                       'RoBERTa-base':[RobertaTokenizer.from_pretrained('roberta-base'),RobertaModel.from_pretrained('roberta-base')]
                       }


    # load data
    # train = pd.read_csv('data/inputs/subtest.csv')
    train = pd.read_csv('./data/subset.csv')
    title = train['job_title']
    text = train['job_description']


    # chooose from DistilBERT, RoBERTa, etc
    selected = 'RoBERTa-base'

    model = language_models[selected][1]
    tokenizer = language_models[selected][0]


    title_embeddings, text_embeddings = generate_embeddings(model, tokenizer, title, text)

    print(title_embeddings)
    


    # save the embeddings
    # torch.save(title_embeddings, f'data/embeddings/embeddings_title_{lang_model.__class__.__name__}.pt')
    # torch.save(text_embeddings, f'data/embeddings/embeddings_text_{lang_model.__class__.__name__}.pt')