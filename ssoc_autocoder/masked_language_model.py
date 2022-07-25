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

# probablity for normal masking (should take note when using normal word masking)

prob = 0.15

# data collator for masking this masks the token 

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)

MASK_TOKEN = tokenizer.mask_token

# choose between normal masking and whole word masking 

type_of_masking = 'normal masking'

# path where the text data set is

path = 'C:/Users/Hari Shiman/Desktop/Data/text/trial.txt'

batch_size = 32

# number of rows of the eval data set based on the fraction of size of training dataset

frac = 0.1

# number of rows of the training data set

train_size = 1000

whole_word_masking_probability = 0.2

sample_text = f"top up {MASK_TOKEN}"

name_of_model = 'ml_model_trial'

def tokenize_function(dataset):

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
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        
    return result




def group_texts(tokenized_dataset):

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


def whole_word_masking_data_collator(features):

    """
    To mask out the occurence of the word in the whole corpus

    Parameters:
        features : contains "input_ids", "attention_mask", "labels",'word_ids'

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

def split_dataset(train_size,fraction,grouped_tokenized_datasets,seed=1):

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

def masking(downsampled_dataset,function,batch_size,split):

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
    tf_dataset = 0 
    # When used normal masking 

    if function ==data_collator:

        tf_dataset = downsampled_dataset[split].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=function,
        shuffle=True,
        batch_size=batch_size,
    )

    # When used whole word masking 

    elif function ==whole_word_masking_data_collator:
        tf_dataset = downsampled_dataset[split].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels",'word_ids'],
        collate_fn=function,
        shuffle=True,
        batch_size=batch_size,
    )

    if tf_dataset==0:
        print(f"Error tf_{split}_dataset is not initialised")
        return

    print(f"\nCompleted masking of tf_{split}_dataset\n")

    return tf_dataset


def get_metrics(model,tf_eval_dataset):

    """
    Getting to know how the base model performs after training 

    Parameters:
        model: the machine learning model
        tf_eval_dataset: tokenized eval_dataset
        


    Returns:
        Perplexiity 
    """
    eval_loss = model.evaluate(tf_eval_dataset)
    
    return math.exp(eval_loss)

def predict_masked_word(text,new_model):
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
    Initializing trainer for training of the model

    Parameters:
        tf_train_dataset: train_dataset
        model: base model without compling the optimizer
        lr: learning rate
        warmup: number of warmup steps
        wdr: weight deacy rate

    Returns:
        The model complied with the optimizer
    """
    num_train_steps = len(tf_train_dataset)
    optimizer, schedule = create_optimizer(
        init_lr=lr,
        num_warmup_steps=warmup,
        num_train_steps=num_train_steps,
        weight_decay_rate=wdr,
    )

    # You need to comple a model to train 

    model.compile(optimizer=optimizer)

    print('\nComplied the optimizer with the model\n')

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    return model
    

def main(model,tokenizer,path,whole_word_masking_probability,train_size,fraction,sample_text,type_of_masking,model_name,batch_size,seed=1,lr = 2e-5,warmup = 1_000,wdr = 0.01):
    """
    Running the whole script

    Parameters:
        model : base model
        tokenizer : tokenizer based on the checkpoint
        path : filepath of the text data
        whole_word_masking_probability : probability for the whole word masking
        train_size : number of rows for train_set you desire
        fraction : ratio against number of rows against train_size you want for eval_dataset
        sample_text : sample text for prediction of the trained model
        type_of_masking : choose between class masking (1) and whole word masking (2)
        model_name : name you want to save the model under
        batch_size : batch_size : number of training examples in one training iteration for the machine learning model
        seed : the random split of the dataset into tf_train_dataset and tf_eval_dataset
        lr: learning rate for machine learning model training
        warmup: number of warmup steps for machine learning model training
        wdr: weight deacy rate for machine learning model training 

    Returns:
        
    """


    #load the text file into appropriate formate

    dataset = load_dataset('text', data_files=path)

    
    # Use batched=True to activate fast multithreading!
    # tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"] )

    print("\nTokenized the dataset\n")

    # Combine the sentences and break into groups of the desired chunk size

    grouped_tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

    print(f'\nGrouped the tokenized dataset into chunks of {chunk_size}\n')

    # Choosing between normal masking and whole word masking 

    if type_of_masking == 'normal masking':
        fn = data_collator

    elif type_of_masking =='whole word masking':
        fn = whole_word_masking_data_collator

    
    # Split the tokenized grouped dataset into train and eval sub datasets

    downsampled_dataset = split_dataset(train_size,fraction,grouped_tokenized_datasets)

    # Random masking of the train set

    tf_train_dataset = masking(downsampled_dataset,fn,batch_size,'train')

    #Random maksing of the eval set

    tf_eval_dataset = masking(downsampled_dataset,fn,batch_size,'test')

    #Compling the base model with the optimizer

    model = trainer(tf_train_dataset,model)

    #Getting perplexity results before training 

    
    print(f"\nPerplexity before training: {(get_metrics(model,tf_eval_dataset)):.2f}\n")

    #Getting perplexity results after training 

    model.fit(tf_train_dataset, validation_data=tf_eval_dataset)

    print(f"\nPerplexity after training: {(get_metrics(model,tf_eval_dataset)):.2f}\n")

    #Save the trained model under the the varibale model_name

    model.save_pretrained(model_name, saved_model=True)

    #Load the saved model

    loaded_model = TFAutoModelForMaskedLM.from_pretrained(model_name)

    #Get the top 5 prediction using the model you trained and a sample text

    predictions = predict_masked_word(sample_text,loaded_model)

    # Print out each prediciton

    for pred in predictions:
        print(f">>> {pred['sequence']}")

    print("completed training")



main(model,tokenizer,path,whole_word_masking_probability,train_size,frac,sample_text,type_of_masking,'trial2',batch_size)