from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_alphabet(dataset_name: str):
    if dataset_name == 'sst':
        return ['æ', 'à', 'é', ' ', '!', '$', '%', '"', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
                '‘', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ö']
    else:
        # Handle other datasets as needed
        return None  # Or raise an error if no alphabet is set for other datasets