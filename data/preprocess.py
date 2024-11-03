from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import random
import torch

def tokenize(batch, tokenizer):
    return tokenizer(batch['sentence'], padding=True, truncation=True)

def tokenized_dataloader(data, tokenizer, batch_size=32, shuffle=True):
    # Apply the tokenize function to the entire dataset
    tokenized_data = data.map(lambda batch: tokenize(batch, tokenizer), batched=True, batch_size=len(data))
    
    # Convert to PyTorch tensors
    tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create DataLoader
    dataloader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def normal_dataloader(data, batch_size=32, shuffle=True):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader