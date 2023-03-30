import functools
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline
from transformers import default_data_collator
from transformers import get_scheduler
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.data.data_collator import tf_default_data_collator

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

import collections
import numpy as np
import math
import argparse

import mlflow
from omegaconf import DictConfig
import re
import hydra

import os

def tokenize_function(dataset,tokenizer):

    """
    Tokenize the sentences

    Parameters:
        dataset : text data that you would want to tokenize

    Returns:
        Tokenized sequences
    """

    result = tokenizer(dataset["text"])
    
    # It is to grab word IDs in the case we are doing whole word masking

    if tokenizer.is_fast:

        #getting the word_ids of the tokens

        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        
    return result




def group_texts(tokenized_dataset,chunk_size):

    """
    Grouping all the text data and splitting into groups based on chunk size

    Parameters:
        tokenized_dataset : tokenized text data that you would want to pass to the model

    Returns:
       Groups of tokenized text data according to the chunk size set
    """
    # Concatenate all texts
    concatenated_tokenized_dataset = {k: sum(tokenized_dataset[k], []) for k in tokenized_dataset.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_tokenized_dataset[list(tokenized_dataset.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_tokenized_dataset.items()
    }
    # Create a new labels column
    # This a copy of the tokens in inputs id as the is the ground truth of the tokens in the sentence
    # The labels will be used as the ground truth when predicting the masked token
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features,whole_word_masking_probability,tokenizer):

    """
    To mask out the occurence of the word in the whole corpus

    Parameters:
        features : contains "input_ids", "attention_mask", "labels",'word_ids'
        whole_word_masking_probability : the probability the how many whole words are masked
        tokenizer : tokenizer of the model we are using


    Returns:
        A function that masks a sequence of tokens with random words masked throughout the whole sequence.
        The occurence of the word will be masked in the whole corpus.
        You can adjust the probabilty of the number of words by settign whole_word_masking_probability
    """
    

    for feature in features:

        # Getting the list of word_ids for each row of data indicating the index of word each token comes from

        word_ids = feature.pop('word_ids')

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, whole_word_masking_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]

        # labels are all -100 except for the ones corresponding to mask words.

        new_labels = [-100] * len(labels)

        # for each word_id that was chosen to be masked 

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()

            # for each token id that comes from the word that was chosen to be masked

            for idx in mapping[word_id]:

                # labels are all -100 except for the ones corresponding to mask words.

                new_labels[idx] = labels[idx]

                # masking of the word in the input

                input_ids[idx] = tokenizer.mask_token_id

    return tf_default_data_collator(features)

def split_dataset(train_size,fraction,grouped_tokenized_datasets,seed):

    """
    Split the dataset into train and eval

    Parameters:
        train_size : number of rows of tokenized sequences 
        fraction : ratio of train_size you want as eval data set
        grouped_tokenized_datasets : grouped tokenized dataset
        seed : to randomize the split of the dataset

    Returns:
        Dataset spilt into subset of train and eval dataset
    """

    test_size = int(fraction * train_size)

    downsampled_dataset = grouped_tokenized_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=seed
    )

    print("\nSplit dataset into train and eval\n")

    return downsampled_dataset

def masking(downsampled_dataset,function,batch_size,split,type_of_masking):

    """
    Masking the train_data set according to the masking technique you want 

    Parameters:
        downsampled_dataset : dataset which has already been split into train and eval dataset
        function : choose between normal masking (1) and whole word masking (2)
        batch_size : number of training examples in one training iteration
        split : either train or eval split

    Returns:
        The whole of trained dataset tokenized
    """
    # tf_dataset = 0 
    # When used normal masking 
    tf_dataset = downsampled_dataset[split].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=function,
    shuffle=True,
    batch_size=batch_size)
    #if type_of_masking == 'normal masking':
#
    #    tf_dataset = downsampled_dataset[split].to_tf_dataset(
    #    columns=["input_ids", "attention_mask", "labels"],
    #    collate_fn=function,
    #    shuffle=True,
    #    batch_size=batch_size,
    #)
#
    ## When used whole word masking 
#
    #elif type_of_masking == 'whole word masking':
    #    tf_dataset = downsampled_dataset[split].to_tf_dataset(
    #    columns=["input_ids", "attention_mask", "labels",'word_ids'],
    #    collate_fn=function,
    #    shuffle=True,
    #    batch_size=batch_size,
    #)

    if tf_dataset==0:
        print(f"Error tf_{split}_dataset is not initialised")
        return

    print(f"\nCompleted masking of tf_{split}_dataset\n")

    return tf_dataset


def insert_random_mask(batch,data_collator):
    """
    Apply the masking once on the whole test set, and then use the default data collator in 
    Transformers to collect the batches during evaluation.

    Parameters:
        batch:
        data_collator: normal masking or whole word masking

    Returns:
        The whole of trained dataset tokenized
    """

    

    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    
def predict_masked_word(text,new_model,tokenizer):
    """
    Getting prediction of the [MASK] token

    Parameters:
        text: text with [MASK] token you want predictions for
        new_model: machine learning model that predicts the mask token

    Returns:
        Top 5 predictions for the [MASK] token
    """

    #load the mask predictor
    
    mask_filler = pipeline(
    "fill-mask", model=new_model,tokenizer=tokenizer)

    #top 5 predictions of the [MASK] token
    
    preds = mask_filler(text)

    return preds

class MyModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        return None

    def predict(self, context, model_input):
        return self.my_custom_function(model_input)

    def my_custom_function(self, model_input):
        # do something with the model input
        return 0

def trainer(model,lr,train_dataloader,eval_dataloader,num_of_epochs,num_warmup_steps,eval_dataset,filename,batch_size,args):

    """
    Initializing trainer for training of the model

    Parameters:
        model: base mlm model
        lr: learning rate
        train_dataloader : training set
        eval_dataloader : eval set
        num_of_epochs : number of epochs
        num_warmup_steps : number of warmup steps
        eval_dataset : eval_datset before it is pushed throught dataloader
        filename : the text file where training results will be stored 


        

    Returns:
        The model which has finished training 
    """
    optimizer = AdamW(model.parameters(), lr=lr)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

    num_train_epochs = num_of_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    )

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    with open(filename,'a',encoding='utf-8') as f:

        f.write(f"/n Before Training , Perplexity: {perplexity} Loss: {torch.mean(losses)}")


    print(f">>> Before Training , Perplexity: {perplexity} Loss: {torch.mean(losses)}")
    writer = SummaryWriter()
    #if any(map(lambda x: bool(re.search(args.experiment_name, x.name)), mlflow.search_experiments())):
    #    experiment_id = mlflow.get_experiment_by_name(args.experiment_name).experiment_id
    #else:
    #    experiment_id = mlflow.create_experiment(args.experiment_name).experiment_id
    progress_bar = tqdm(range(num_training_steps), position=0, leave=True)
    #with mlflow.start_run(experiment_id=experiment_id):
    with mlflow.start_run():
    #with mlflow.start_run():
        for epoch in range(num_train_epochs):
            # Training
            training_loss = []
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                training_loss.append(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            mlflow.log_metric(key = 'Training loss', value = training_loss[0].item(), step = epoch)
            # Evaluation
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            writer.add_scalar("Loss/train", torch.mean(losses), epoch)
            try:
                perplexity = math.exp(torch.mean(losses))
                writer.add_scalar("Perplexity/train", perplexity, epoch)
            except OverflowError:
                perplexity = float("inf")

            with open(filename,'a',encoding='utf-8') as f:

                f.write(f"/n Epoch {epoch+1}: Perplexity: {perplexity} Loss: {torch.mean(losses)}")
            #metrics = {'Perplexity': perplexity, 'Losses': torch.mean(losses)}
            mlflow.log_metric(key = 'Perplexity', value = perplexity, step = epoch)
            mlflow.log_metric(key = 'Validation loss', value = torch.mean(losses), step = epoch)
            #print(f">>> Epoch {epoch+1}: Perplexity: {perplexity} Loss: {torch.mean(losses)}")
        
        mlflow.log_param('epoch', args.epochs)
        mlflow.log_param('normal_masking_probability', args.normal_masking_probability)
        mlflow.log_param('warmup', args.warmup)

        pymodel = MyModel(model)
        model_info = mlflow.pyfunc.log_model(artifact_path="testmodel", python_model=pymodel, registered_model_name=args.model.name)




    print('\nFinished Training\n')

    

    

    return model
    
#@hydra.main(config_path="../config/", config_name="config.yaml", version_base = "1.2")
def main(cfg: DictConfig):
    

    
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type = str,
        help ='the base model',
        default = "C:/Users/Hari Shiman/Documents/BERT_models_distilroberta"
    )

    parser.add_argument(
        '--filepath',
        type = str,
        help ='the path where the text file for training is stored in',
        default = 'C:/Users/Hari Shiman/Desktop/Data/text/raw_2018-06.txt'
    )

    parser.add_argument(
        '--normal_masking_probability',
        type = float,
        help ='the probability of words being masked for normal masking',
        default = 0.15
    )

    parser.add_argument(
        '--whole_word_masking_probability',
        type = float,
        help ='the probability of words being masked for whole word masking',
        default = 0.15
    )

    parser.add_argument(
        '--chunk_size',
        type = int,
        help ='the number of tokens in each group the tokenized data will be split into',
        default = 128
    )

    parser.add_argument(
        '--train_size',
        type = int,
        help ='the number of rows used for training',
        default = 100
    )

    parser.add_argument(
        '--fraction',
        type = float,
        help ='the fraction of number of training rows to be used for testing',
        default = 0.1
    )

    parser.add_argument(
        '--sample_text',
        type = str,
        help ='a sample text used to test our trained model',
        default = "tech startup is"
    )

    parser.add_argument(
        '--type_of_masking',
        type = str,
        help ='choose between "normal masking" or "whole word masking"',
        default = "normal masking"
    )

    parser.add_argument(
        '--model_name',
        type = str,
        help ='path/file trained model will be saved as',
        default = 'test1'
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        help ='batch size for the training of the model',
        default = 32
    )

    parser.add_argument(
        '--seed',
        type = int,
        help ='random seed of the splitting of the dataset',
        default = 1
    )

    parser.add_argument(
        '--lr',
        type = float,
        help ='learning rate for the training of the model',
        default = 5e-5
    )

    parser.add_argument(
        '--warmup',
        type = int,
        help ='number of warmup steps for the training of the model',
        default = 0
    )
    

    parser.add_argument(
        '--wdr',
        type = float,
        help ='weight decay rate for the training of the model',
        default = 0
    )

    parser.add_argument(
        '--epochs',
        type = int,
        help ='number of epochs for the training of the model',
        default = 3
    )

    parser.add_argument(
        '--filename',
        type = str,
        help ='same of file where training logs will be storedd',
        default = 'logs.txt'
    )

    args = parser.parse_args()
    '''
    # just testing/ will modify the structure once the test works!
    '''
    args.lr = cfg.lr
    args.model = cfg.model
    args.filepath = cfg.filepath
    args.train_size = cfg.train_size
    args.fraction = cfg.fraction
    args.epochs = cfg.epochs
    args.normal_masking_probability = cfg.normal_masking_probability
    args.warmup = cfg.warmup
    print(os.getcwd())
    '''
    args = cfg
    model = AutoModelForMaskedLM.from_pretrained(args.model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_path)

    #load the text file into appropriate formate

    dataset = load_dataset('text', data_files=args.dataset.filepath)
    
    
    # Use batched=True to activate fast multithreading!
    # tokenize the dataset
    tokenize = functools.partial(tokenize_function, tokenizer=tokenizer)
    tokenized_datasets = dataset.map(
        tokenize,batched=True, remove_columns=["text"] )

    print("\nTokenized the dataset\n")

    # Combine the sentences and break into groups of the desired chunk size
    group = functools.partial(group_texts, chunk_size=args.chunk_size)
    grouped_tokenized_datasets = tokenized_datasets.map(group, batched=True)

    print(f'\nGrouped the tokenized dataset into chunks of {args.chunk_size}\n')

    # Split the tokenized grouped dataset into train and eval sub datasets

    downsampled_dataset = split_dataset(args.train_size,args.fraction,grouped_tokenized_datasets,args.seed)

    # Choosing between normal masking and whole word masking 

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.normal_masking_probability)
    fn = data_collator
    #if args.type_of_masking == 'normal masking':
    #    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.normal_masking_probability)
    #    fn = data_collator
        

    #elif args.type_of_masking =='whole word masking':
    #    fn = functools.partial(whole_word_masking_data_collator, whole_word_masking_probability=args.whole_word_masking_probability,tokenizer=tokenizer)

    insert_random = functools.partial(insert_random_mask, data_collator=fn)

    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    eval_dataset = downsampled_dataset["test"].map(
    insert_random,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,)

    eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

    
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
    )


    #Evaluate and train the model 

    model = trainer(model,args.lr,train_dataloader,eval_dataloader,args.epochs,args.warmup,eval_dataset,args.filename,args.batch_size,args)


    ##Save the trained model under the the varibale model_name
#
    #model.save_pretrained(args.model_name, saved_model=True)
#
    ##Load the saved model
#
    #loaded_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
#
    ##Get the top 5 prediction using the model you trained and a sample text
#
    #MASK_TOKEN = tokenizer.mask_token
    #sample_text = f'{args.sample_text} {MASK_TOKEN}'
#
    #predictions = predict_masked_word(sample_text,loaded_model,tokenizer)
#
    ## Print out each prediciton
#
    #for pred in predictions:
    #    print(f">>> {pred['sequence']}")
#
    #print("completed script")


if __name__ == "__main__":
    
    main()