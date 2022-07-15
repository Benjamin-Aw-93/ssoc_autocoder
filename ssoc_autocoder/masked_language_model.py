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




def choose_model(model_checkpoint):
    
    model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)

    return model

def choose_tokenizer(model_checkpoint):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return tokenizer


model_checkpoint = "distilroberta-base"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
chunk_size =128
wwm_probability = 0.2
prob = 0.15

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)
MASK_TOKEN = tokenizer.mask_token


def load_file(path):

    dataset = load_dataset('text', data_files=path)

    return dataset

def tokenize_function(examples):

    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        
    return result


# Use batched=True to activate fast multithreading!

def group_texts(examples):
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
    train_size = 10000
    test_size = int(fraction * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    return downsampled_dataset

def masking_train(downsampled_dataset,function):
    if function ==1:

        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )
    elif function ==2:
        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels",'word_ids'],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )

    return tf_train_dataset


def masking_eval(downsampled_dataset,function):

    if function ==1:

        tf_eval_dataset = downsampled_dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=function,
        shuffle=False,
        batch_size=32,
    )

    elif function ==2:
        tf_eval_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels",'word_ids'],
        collate_fn=function,
        shuffle=True,
        batch_size=32,
    )

    return tf_eval_dataset



def pre_eval(model,tf_eval_dataset):
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity before training: {math.exp(eval_loss):.2f}")
    return math.exp(eval_loss)

def get_metrics(model,tf_eval_dataset,tf_train_dataset):
    model.fit(tf_train_dataset, validation_data=tf_eval_dataset)
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity after training : {math.exp(eval_loss):.2f}")
    return math.exp(eval_loss)

def sample_predictions(text,new_model):
    
    mask_filler = pipeline(
    "fill-mask", model=new_model,tokenizer=tokenizer)
    
    preds = mask_filler(text)

    return preds

def trainer(tf_train_dataset,model,lr = 2e-5,warmup = 1_000,wdr = 0.01):
    num_train_steps = len(tf_train_dataset)
    optimizer, schedule = create_optimizer(
        init_lr=lr,
        num_warmup_steps=warmup,
        num_train_steps=num_train_steps,
        weight_decay_rate=wdr,
    )
    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    return model

def mlm(prob=0.15):


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    return data_collator
    


def main(model,tokenizer,path,wwm_probability,train_size,fraction,text,function,model_name):

    model = model

    tokenizer = tokenizer

    dataset = load_file(path)


    # Use batched=True to activate fast multithreading!
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"] )

    


    lm_datasets = tokenized_datasets.map(group_texts, batched=True)


    wwm_probability = wwm_probability

    if function == 1:
        fn = data_collator

    elif function ==2:
        fn = whole_word_masking_data_collator

    

    downloaded_dataset = split_dataset(train_size,fraction,lm_datasets)

    tf_train_set = masking_train(downloaded_dataset,fn)

    tf_eval_set = masking_eval(downloaded_dataset,fn)

    model = trainer(tf_train_set,model)

    pre_eval(model,tf_eval_set)

    get_metrics(model,tf_eval_set,tf_train_set)


    model.save_pretrained(model_name, saved_model=True)
    
    new_model = TFAutoModelForMaskedLM.from_pretrained(model_name)

    predictions = sample_predictions(text,new_model)

    for pred in predictions:
        print(f">>> {pred['sequence']}")

    print("completed training")

main(model,tokenizer,'C:/Users/Hari Shiman/Desktop/Data/text/trial.txt',0.2,10000,0.1,f"top up {MASK_TOKEN}",1,'trial2')