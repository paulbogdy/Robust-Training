import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from models.model_wrapper import ModelWrapper
import pandas as pd
import random

def perturb_sentence(sentence, alphabet, q=5, custom_char="§"):
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N / 100)
    nr_insertions = int(nr_perturbations / 3)
    nr_deletions = int(nr_perturbations / 3)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        return random.choice(alphabet)

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

class RandCharV7Trainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        """
        Initialize the BaseTrainer with model, device, optimizer, and other configurations.
        """
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.q = args.q
        self.alpha = args.alpha
        self.pert_mode = args.pert_mode

        self.base_path = f'rand_char_v7_{args.pert_mode}_q{self.q}_a{str(self.alpha).replace(".", "_")}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--alpha', type=float, default=2, help='The param for the beta distribution')
        parser.add_argument('--pert_mode', type=str, choices=['aug', 'replace'], default='aug', help='How to add the perturbations to the batch, aug means append the perturbed sentences to the batch, replace means replace the original sentences with the perturbed ones')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        # Total steps for learning rate scheduling
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        # Learning rate schedulers
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Training over epochs
        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))
            
            for batch_idx, batch in pbar:
                # Extract data and move to device
                inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                # Perturb inputs
                perturbed_inputs = [perturb_sentence(sentence, self.alphabet, q=self.q) for sentence in inputs]

                if (self.pert_mode == 'aug'):
                    inputs += perturbed_inputs
                else:
                    inputs = perturbed_inputs

                # Tokenize inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                input_embeds = self.model.input_embeddings(input_ids)
                batch_perm = torch.randperm(input_embeds.size(0))

                shuffled_input_embeds = input_embeds[batch_perm]

                soft_original_labels = torch.nn.functional.one_hot(labels, num_classes=self.model.num_labels()).float()
                soft_shuffled_labels = soft_original_labels[batch_perm]

                lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().to(self.device)

                # interpolate the [CLS] token embeddings
                mixed_input_embeds = lam * input_embeds + (1 - lam) * shuffled_input_embeds
                mixed_labels = lam * soft_original_labels + (1 - lam) * soft_shuffled_labels

                mixed_outputs = self.model.forward_embeddings(input_embeds=mixed_input_embeds, attention_mask=attention_mask)
                loss_mix = F.kl_div(F.log_softmax(mixed_outputs.logits, dim=-1), mixed_labels, reduction='batchmean')

                outputs = self.model.forward_embeddings(input_embeds=input_embeds, attention_mask=attention_mask)
                loss_ce = self.loss_fn(outputs.logits, labels)

                loss = loss_ce + loss_mix

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update scheduler
                scheduler.step()

                # Logging loss
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation step
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
