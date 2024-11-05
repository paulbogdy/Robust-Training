from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm

class ModelWrapper:
    def __init__(self, model_name: str, num_labels: int):

        if model_name == 'bert':
            model = 'bert-base-uncased'
        elif model_name == 'roberta':
            model = 'roberta-base'
        elif model_name == 'albert':
            model = 'albert-base-v2'
        else:
            raise ValueError(f"Model {model_name} not recognized")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels)

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def tokenize(self, texts):
        """
        Tokenizes a batch of texts.
        Args:
            texts: List of input texts to tokenize.
        Returns:
            tokenized: Tokenized batch ready for model input.
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        Args:
            inputs: Input IDs from tokenizer
            attention_mask: Attention mask if needed
        Returns:
            output: Model output, typically logits
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def forward_embeddings(self, input_embeds, attention_mask=None):
        """
        Forward pass of the model using input embeddings.
        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask if needed
        Returns:
            output: Model output, typically logits
        """
        return self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    
    def input_embeddings(self, inputs):
        """
        Computes input embeddings.
        Args:
            inputs: Input IDs from tokenizer
        Returns:
            embeddings: Model input embeddings
        """
        return self.model.base_model.embeddings.word_embeddings(inputs)

    def adversarial_forward(self, inputs, attack_fn, **kwargs):
        """
        Runs adversarial forward pass using a specified attack function.
        Args:
            inputs: Original input IDs
            attack_fn: Function that applies adversarial perturbations
        Returns:
            perturbed_output: Model output on adversarially perturbed inputs
        """
        perturbed_inputs = attack_fn(inputs, **kwargs)
        return self.forward(**perturbed_inputs)

    def evaluate(self, data_loader, device):
        """
        Evaluates model performance on a data loader.
        Args:
            data_loader: Dataloader for evaluation
            device: Device to run model on (CPU or GPU)
        Returns:
            accuracy: Evaluation accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                inputs = batch['sentence']
                labels = batch['label'].to(device)

                # Tokenize inputs
                tokenized_inputs = self.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(device)
                attention_mask = tokenized_inputs['attention_mask'].to(device)

                outputs = self.forward(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
    
        if total == 0:
            return 0.0  # Handle case where there are no samples

        return correct / total

    def save(self, save_path):
        """Saves the model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path):
        """Loads a model and tokenizer from specified path."""
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
