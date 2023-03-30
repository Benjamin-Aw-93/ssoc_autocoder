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