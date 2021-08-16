# Importing required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Setting up the device if GPU is available
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# Try to get summarised data out using HTML tags

mcf_df = pd.read_csv("..\Data\Processed\WGS_Dataset_JobInfo_precleaned.csv")

mcf_df
