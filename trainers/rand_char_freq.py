import os
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper
import random

def perturb_sentence(sentence, alphabet, alphabet_distribution, q=5, custom_char="§"):
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N / 100)
    nr_insertions = int(nr_perturbations / 3)
    nr_deletions = int(nr_perturbations / 3)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        # Sample from the alphabet distribution
        return random.choices(alphabet, weights=alphabet_distribution, k=1)[0]

    # Perform insertions
    insert_positions = random.sample(range(N), nr_insertions)
    for pos in insert_positions:
        new_char = get_random_char()
        modified_pos = 2 * pos + 1
        perturbed_sentence = perturbed_sentence[:modified_pos] + new_char + perturbed_sentence[modified_pos + 1:]
    
    # Perform deletions
    delete_positions = random.sample(range(N), nr_deletions)
    for pos in delete_positions:
        modified_pos = 2 * pos
        perturbed_sentence = perturbed_sentence[:modified_pos] + custom_char + perturbed_sentence[modified_pos + 1:]

    # Perform swaps
    swap_positions = random.sample(range(N), nr_swaps)
    for pos in swap_positions:
        new_char = get_random_char()
        while new_char == perturbed_sentence[2 * pos]:
            new_char = get_random_char()
        modified_pos = 2 * pos
        perturbed_sentence = perturbed_sentence[:modified_pos] + new_char + perturbed_sentence[modified_pos + 1:]
    
    return ''.join(char if char != custom_char else '' for char in perturbed_sentence)

class RandCharFreqTrainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alphabet = alphabet
        self.q = args.q
        self.r = args.r
        
        self.base_path = f'rand_char_freq_q{self.q}_r{str(self.r).replace(".", "_")}'
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--r', type=float, default=1, help='Rate between the character with maximum probability and the character with minimum probability')
        return parser

    def compute_alphabet_distribution(self, train_loader):
        print("Computing alphabet distribution...")
        
        # Initialize frequency dictionary
        freq = {char: 0 for char in self.alphabet}
        
        # Compute raw frequencies
        for batch in tqdm(train_loader):
            for sentence in batch['sentence']:
                for char in sentence:
                    if char in freq:
                        freq[char] += 1

        # Convert frequencies to probabilities using the log-inverse formula
        raw_prob = {}
        for char, count in freq.items():
            raw_prob[char] = 1 / (np.log(count + 2))  # Avoid division by zero with + 2

        # Compute current max-min ratio
        max_raw = max(raw_prob.values())
        min_raw = min(raw_prob.values())
        r_raw = max_raw / min_raw

        # Compute scaling factor alpha
        alpha = np.log(self.r) / np.log(r_raw)

        # Scale raw probabilities
        scaled_prob = {char: prob ** alpha for char, prob in raw_prob.items()}

        # Normalize probabilities
        total = sum(scaled_prob.values())
        final_distribution = {char: prob / total for char, prob in scaled_prob.items()}

        final_weights = [final_distribution[char] for char in self.alphabet]
        
        print("Alphabet distribution computed.")
        return final_weights

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with random character perturbations.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        alphabet_distribution = self.compute_alphabet_distribution(train_loader)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                # Perturb inputs
                inputs = [perturb_sentence(sentence, self.alphabet, alphabet_distribution, q=self.q) for sentence in batch['sentence']]
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
