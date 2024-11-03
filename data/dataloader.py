from datasets import load_dataset

def load_dataset(dataset_name: str):
    if dataset_name == 'sst':
        dataset = load_dataset('glue', 'sst2')
        dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label']})
    elif dataset_name == 'imdb':
        dataset = load_dataset('imdb')
        dataset = dataset.map(lambda x: {'sentence': x['text'], 'label': x['label']})
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset