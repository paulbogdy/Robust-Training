import datasets

def load_dataset(dataset_name: str):
    if dataset_name == 'sst':
        dataset = datasets.load_dataset('glue', 'sst2')
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")