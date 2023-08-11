from transformers import DistilBertTokenizer, AutoModel
import torch
import pandas as pd
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
    
    title_embeddings = embeddings_iterator(title_loader)
    text_embeddings = embeddings_iterator(text_loader)
    
    return title_embeddings, text_embeddings

if __name__ == "__main__":
    # load artifacts, data, then generate and store embeddings
    lang_model_path = 'artifacts/lang-model/'
    tokenizer_path = 'artifacts/distilbert-tokenizer-pretrained-7epoch'
    lang_model = AutoModel.from_pretrained(lang_model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    # load data
    train = pd.read_csv('data/inputs/subtest.csv')
    title = train['job_title']
    text = train['job_description']

    title_embeddings, text_embeddings = generate_embeddings(lang_model, tokenizer, title, text)
    # save the embeddings
    torch.save(title_embeddings, f'data/embeddings/embeddings_title_{lang_model.__class__.__name__}.pt')
    torch.save(text_embeddings, f'data/embeddings/embeddings_text_{lang_model.__class__.__name__}.pt')