import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper
import random


def perturb_sentence_with_mask(sentence, mask_token="[MASK]", q=5):
    """
    Masks a percentage of words in the sentence instead of randomly changing characters.
    
    Args:
    - sentence (str): The input sentence.
    - mask_token (str): Token to use for masking words.
    - q (int): Percentage of words to mask.
    
    Returns:
    - str: Sentence with some words masked.
    """
    words = sentence.split()
    N = len(words)
    nr_masks = max(1, int(q * N / 100))  # Number of words to mask

    # Choose random positions to mask
    mask_positions = random.sample(range(N), nr_masks)
    for pos in mask_positions:
        words[pos] = mask_token

    return ' '.join(words)


class RandMaskTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.q = args.q  # Percentage of words to mask
        self.base_path = f'rand_mask_q{self.q}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Masking rate (0-100) representing percentage of words to mask')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with random word masking.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                # Perturb inputs with word masking
                inputs = [perturb_sentence_with_mask(sentence, q=self.q) for sentence in batch['sentence']]
                labels = batch['label'].to(self.device)

                # Tokenize perturbed inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)

                # Gradient clipping and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update scheduler
                scheduler.step()

                # Log batch loss
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Epoch logging
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
