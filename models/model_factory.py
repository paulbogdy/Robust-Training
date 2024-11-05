from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_wrapper import ModelWrapper

def load_model_and_tokenizer(model_name: str, num_labels: int):
    """
    Load a pre-trained model for sequence classification.
    Args:
        model_name: The model name
        num_labels: The number of labels in the classification task
    Returns:
        model: The pre-trained model
        tokenizer: The pre-trained tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return model, tokenizer

def load_wrapper(model_name: str, num_labels: int):
    """
    Load a model wrapper for sequence classification.
    Args:
        model_name: The model name
        num_labels: The number of labels in the classification task
    Returns:
        model: The model wrapper
    """

    return ModelWrapper(model_name, num_labels)