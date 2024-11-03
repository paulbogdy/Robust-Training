import datasets

def load_dataset(dataset_name: str):
    if dataset_name == 'sst':
        dataset = datasets.load_dataset('glue', 'sst2')
        dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label']})
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset