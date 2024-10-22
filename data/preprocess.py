from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def tokenize(batch, tokenizer):
    return tokenizer(batch['sentence'], padding=True, truncation=True)

def get_dataloader(data, tokenizer, batch_size, shuffle=True):
    # Apply the tokenize function to the entire dataset
    tokenized_data = data.map(lambda batch: tokenize(batch, tokenizer), batched=True, batch_size=len(data))
    
    # Convert to PyTorch tensors
    tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create DataLoader
    dataloader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
