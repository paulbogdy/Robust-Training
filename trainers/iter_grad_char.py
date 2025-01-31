import os
import torch
import random
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper


def map_gradients_to_characters(sentence, token_gradients, token_offsets):
    """
    Map token-level gradient scores to character-level probabilities.
    """
    char_scores = [0] * len(sentence)

    # Assign token gradient scores to character positions
    for token_idx, (start, end) in enumerate(token_offsets):
        token_score = token_gradients[token_idx]
        for char_idx in range(start, end):
            char_scores[char_idx] += token_score / (end - start)  

    # Handle spaces
    for char_idx, char in enumerate(sentence):
        if char == " ":
            left_score = char_scores[char_idx - 1] if char_idx > 0 else 0
            right_score = char_scores[char_idx + 1] if char_idx < len(sentence) - 1 else 0
            char_scores[char_idx] = (left_score + right_score) / 2

    # Normalize
    total_score = sum(char_scores)
    if total_score > 0:
        char_scores = [score / total_score for score in char_scores]

    return char_scores


def perturb_sentence_with_distribution(sentence, distribution, alphabet, q=5, custom_char="§"):
    """
    Apply character-level perturbations based on importance distribution.
    """
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N / 100)
    nr_insertions = int(nr_perturbations / 3)
    nr_deletions = int(nr_perturbations / 3)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        return random.choice(alphabet)

    insert_positions = random.choices(range(len(sentence)), weights=distribution, k=nr_insertions)
    other_positions = random.choices(range(len(sentence)), weights=distribution, k=nr_deletions + nr_swaps)

    # Insert characters
    for pos in insert_positions:
        new_char = get_random_char()
        perturbed_sentence = perturbed_sentence[:2 * pos + 1] + new_char + perturbed_sentence[2 * pos + 2:]

    # Delete characters
    for pos in other_positions[:nr_deletions]:
        perturbed_sentence = perturbed_sentence[:2 * pos] + custom_char + perturbed_sentence[2 * pos + 1:]

    # Swap characters
    for pos in other_positions[nr_deletions:]:
        new_char = get_random_char()
        while new_char == perturbed_sentence[2 * pos]:
            new_char = get_random_char()
        perturbed_sentence = perturbed_sentence[:2 * pos] + new_char + perturbed_sentence[2 * pos + 1:]

    return ''.join(char if char != custom_char else '' for char in perturbed_sentence)


class IterativeGradCharTrainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alphabet = alphabet
        self.q = args.q
        self.nr_iter = args.nr_iter  # Number of iterations for perturbation

        self.base_path = f'iter_grad_char_q{self.q}_iter{self.nr_iter}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--nr_iter', type=int, default=4, help='Number of iterations for iterative gradient perturbations')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Training loop with iterative gradient-based character perturbations.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)

        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                for iter_idx in range(self.nr_iter):  # 🔁 Apply perturbation iteratively
                    # Tokenize inputs
                    tokenized_inputs = self.model.tokenize(inputs, return_offsets_mapping=True)
                    input_ids = tokenized_inputs['input_ids'].to(self.device)
                    attention_mask = tokenized_inputs['attention_mask'].to(self.device)
                    offsets = tokenized_inputs['offset_mapping']

                    # Compute embeddings and gradients
                    embeddings = self.model.input_embeddings(input_ids).detach().clone().requires_grad_(True)
                    outputs = self.model.forward_embeddings(embeddings, attention_mask=attention_mask)
                    loss = self.loss_fn(outputs.logits, labels)
                    loss.backward()

                    # Compute token-level gradient magnitudes
                    token_gradients = embeddings.grad.norm(dim=-1)

                    # Convert token gradients to character-level importance
                    char_distributions = [
                        map_gradients_to_characters(sentence, token_gradients[i], offsets[i])
                        for i, sentence in enumerate(inputs)
                    ]

                    # Perturb sentences iteratively (apply perturbations gradually)
                    perturbation_rate = self.q / self.nr_iter  # 🔥 Apply smaller perturbations each time
                    inputs = [
                        perturb_sentence_with_distribution(sentence, char_distributions[i], self.alphabet, q=perturbation_rate)
                        for i, sentence in enumerate(inputs)
                    ]

                # Tokenize final perturbed inputs
                tokenized_perturbed_inputs = self.model.tokenize(inputs)
                input_ids_perturbed = tokenized_perturbed_inputs['input_ids'].to(self.device)
                attention_mask_perturbed = tokenized_perturbed_inputs['attention_mask'].to(self.device)

                self.optimizer.zero_grad()
                outputs_perturbed = self.model.forward(input_ids=input_ids_perturbed,
                                                       attention_mask=attention_mask_perturbed)
                loss = self.loss_fn(outputs_perturbed.logits, labels)

                # Backprop & optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Log batch loss
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Epoch logging
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
