from transformers import FNetForPreTraining, TFAutoModelForMaskedLM, default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
from transformers import create_optimizer
import tensorflow as tf
import math
from transformers.data.data_collator import tf_default_data_collator
from transformers import pipeline
from transformers import pipeline
from tensorflow import keras





# selecting the base model

model_checkpoint = "distilroberta-base"

# selecting the base masked language model based on the base model

model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)

# selecting the tokenizer based on the base model

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# chunk size of the sequence of each group after grouping 

chunk_size = 128

# whole word masking probability (should take note when using whole word masking)

wwm_probability = 0.2

# probablity for normal masking (should take note when using normal word masking)

prob = 0.15

# data collator for masking this masks the token 

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)

MASK_TOKEN = tokenizer.mask_token


def load_file(path):

    """
    Load the text files into an appropriate format to run the model on it

    Parameters:
        path : file path where the text file used as an input is saved

    Returns:
        The text file converted into a dataset format which can be used for the model
    """

    dataset = load_dataset('text', data_files=path)

    return dataset

def tokenize_function(examples):

    """
    Tokenize the sentences

    Parameters:
        examples : text data that you would want to tokenize

    Returns:
        Tokenized sequences
    """

    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        
    return result


# Use batched=True to activate fast multithreading!

def group_texts(examples):

    """
    Grouping all the text data and splitting into groups based on chunk size

    Parameters:
        examples : tokenized text data that you would want to pass to the model

    Returns:
       Groups of text data according to the chunk size set
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features):

    """
    To mask out the occurence of the word in the whole corpus

    Parameters:
        features: contians "input_ids", "attention_mask", "labels"

    Returns:
        A function that masks a sequence of tokens with random words masked throughout the whole sequence.
        The occurence of the word will be masked in the whole corpus.
        You can adjust the probabilty of the number of words by settign wwm_probability
    """
    
    for feature in features:
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
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return tf_default_data_collator(features)

def split_dataset(train_size,fraction,lm_datasets):

    """
    Split the dataset into train and eval

    Parameters:
        train_size : number of rows of tokenized sequences 
        fraction : ratio of train_size you want as eval data set
        lm_datasets: grouped tokenized dataset

    Returns:
        Dataset spilt into subset of train and eval dataset
    """
    train_size = 10000
    test_size = int(fraction * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    print("Split dataset into train and eval")

    return downsampled_dataset

def masking_train(downsampled_dataset,function):

    """
    Masking the train_data set according to the massking technique you want 

    Parameters:
        downsampled_dataset: dataset which has already been split into train and eval dataset
        function: choose between normal masking (1) and whole word masking (2)

    Returns:
        The whole of trained dataset tokenized
    """
    tf_train_dataset = 0 
    # When used normal masking 

    if function ==data_collator:

        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )

    # When used whole word masking 

    elif function ==whole_word_masking_data_collator:
        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels",'word_ids'],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )

    if tf_train_dataset==0:
        print("Error tf_train_dataset is not initialised")
        return

    print("Completed masking of train_dataset")

    return tf_train_dataset


def masking_eval(downsampled_dataset,function):

    """
    Masking the eval_data set according to the masking technique you want 

    Parameters:
        downsampled_dataset: dataset which has already been split into train and eval dataset
        function: choose between normal masking (1) and whole word masking (2)

    Returns:
        The whole of eval dataset tokenized
    """
    tf_eval_dataset = 0

    # When used normal masking 

    if function ==data_collator:

        tf_eval_dataset = downsampled_dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=function,
        shuffle=False,
        batch_size=32,
    )

    # When used whole word masking 

    elif function ==whole_word_masking_data_collator:
        tf_eval_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels",'word_ids'],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )

    if tf_eval_dataset==0:
        print("Error tf_eval_dataset is not initialised")
        return
    

    print("Completed masking of eval_dataset")

    return tf_eval_dataset



def pre_eval(model,tf_eval_dataset):

    """
    Getting to know how the base model performs without training 

    Parameters:
        model: the machine learning model
        tf_eval_dataset: tokenized eval_dataset

    Returns:
        Perplexity before training 
    """
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity before training: {math.exp(eval_loss):.2f}")
    return math.exp(eval_loss)

def get_metrics(model,tf_eval_dataset,tf_train_dataset):

    """
    Getting to know how the base model performs after training 

    Parameters:
        model: the machine learning model
        tf_eval_dataset: tokenized eval_dataset
        tf_train_dataset: tokenized train_dataset


    Returns:
        Perplexiity before training 
    """

    model.fit(tf_train_dataset, validation_data=tf_eval_dataset)
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity after training : {math.exp(eval_loss):.2f}")
    return math.exp(eval_loss)

def sample_predictions(text,new_model):
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

def trainer(tf_train_dataset,model,lr = 2e-5,warmup = 1_000,wdr = 0.01):

    """
    Initilazing trainer for training of the model

    Parameters:
        tf_train_dataset: train_dataset
        model: base model without compling the optimizer
        lr: learning rate
        warmup: number of warmup steps
        wdr: weight deacy rate

    Returns:
        A dictionary of extracted infomation, and the date of the post
    """
    num_train_steps = len(tf_train_dataset)
    optimizer, schedule = create_optimizer(
        init_lr=lr,
        num_warmup_steps=warmup,
        num_train_steps=num_train_steps,
        weight_decay_rate=wdr,
    )
    model.compile(optimizer=optimizer)

    print('Complied the optimizer with the model')

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    return model
    

def main(model,tokenizer,path,wwm_probability,train_size,fraction,text,function,model_name):
    """
    Running the whole script

    Parameters:
        model: base model
        tokenizer: tokenizer based on the checkpoint
        path: filepath of the text data
        wwm_probabilty: probability for the whole word masking
        train_size: number of rows for train_set you desire
        fraction: ratio against number of rows against train_size you want for eval_dataset
        text: sample text for prediction of the trained model
        fucntion: choose between class masking (1) and whole word masking (2)
        model_name: name you want to save the model under

    Returns:
        
    """

    #intilazing the base model based on the model checkpoint

    model = model

    #initalized the tokenizer based on the model checkpoint 

    tokenizer = tokenizer

    #load the text file into appropriate formate

    dataset = load_file(path)


    # Use batched=True to activate fast multithreading!
    # tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"] )

    

    # Combine the sentences and break into groups of the desired chunk size

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # Intialized the whole word masking probability set above

    wwm_probability = wwm_probability

    # Choosing between normal masking(1) and whole word masking (2)

    if function == 1:
        fn = data_collator

    elif function ==2:
        fn = whole_word_masking_data_collator

    
    # Split the tokenized grouped dataset into train and eval sub datasets

    downloaded_dataset = split_dataset(train_size,fraction,lm_datasets)

    # Random masking of the train set

    tf_train_set = masking_train(downloaded_dataset,fn)

    #Random maksing of the eval set

    tf_eval_set = masking_eval(downloaded_dataset,fn)

    #Compling the base model with the optimizer

    model = trainer(tf_train_set,model)

    #Getting perplexity results before training 

    pre_eval(model,tf_eval_set)

    #Getting perplexity results after training 

    get_metrics(model,tf_eval_set,tf_train_set)

    #Save the trained model under the the varibale model_name

    model.save_pretrained(model_name, saved_model=True)\

    #Load the saved model

    new_model = TFAutoModelForMaskedLM.from_pretrained(model_name)

    #Get the top 5 prediction using the model you trained and a sample text

    predictions = sample_predictions(text,new_model)

    # Print out each prediciton

    for pred in predictions:
        print(f">>> {pred['sequence']}")

    print("completed training")

main(model,tokenizer,'C:/Users/Hari Shiman/Desktop/Data/text/trial.txt',0.2,10000,0.1,f"top up {MASK_TOKEN}",1,'trial2')