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

    Args:
        sentence (str): The input sentence.
        token_gradients (list): Importance scores for each token (length = number of tokens).
        token_offsets (list of tuples): List of (start, end) character positions for each token.

    Returns:
        list: Character-level importance scores (length = number of characters in sentence).
    """
    char_scores = [0] * len(sentence)  # Initialize scores for each character

    # Assign token gradient scores to character positions
    for token_idx, (start, end) in enumerate(token_offsets):
        token_score = token_gradients[token_idx]
        for char_idx in range(start, end):  # Distribute score to all characters in the token span
            char_scores[char_idx] += token_score / (end - start)  # Equal distribution across characters

    # Handle spaces
    for char_idx, char in enumerate(sentence):
        if char == " ":
            # Average score of neighboring characters
            left_score = char_scores[char_idx - 1] if char_idx > 0 else 0
            right_score = char_scores[char_idx + 1] if char_idx < len(sentence) - 1 else 0
            char_scores[char_idx] = (left_score + right_score) / 2

    # Normalize to create a probability distribution
    total_score = sum(char_scores)
    if total_score > 0:
        char_scores = [score / total_score for score in char_scores]

    return char_scores


def perturb_sentence_with_distribution(sentence, distribution, alphabet, q=5, custom_char="§"):
    """
    Perturb a sentence based on a given importance distribution.

    Args:
        sentence (str): The input sentence.
        distribution (list): Importance probabilities for each token position.
        alphabet (list): List of possible characters for perturbation.
        q (int): Perturbation rate in percentage.
        custom_char (str): Custom character to aid in insertion/deletion.

    Returns:
        str: The perturbed sentence.
    """
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N / 100)
    nr_insertions = int(nr_perturbations / 3)
    nr_deletions = int(nr_perturbations / 3)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        return random.choice(alphabet)

    # Sample positions based on distribution
    insert_positions = random.choices(range(len(sentence)), weights=distribution, k=nr_insertions)
    other_positions = random.choices(range(len(sentence)), weights=distribution, k=nr_deletions + nr_swaps)

    # Apply perturbations
    for pos in insert_positions:
        new_char = get_random_char()
        perturbed_sentence = perturbed_sentence[:2 * pos + 1] + new_char + perturbed_sentence[2 * pos + 2:]

    for pos in other_positions[:nr_deletions]:
        perturbed_sentence = perturbed_sentence[:2 * pos] + custom_char + perturbed_sentence[2 * pos + 1:]

    for pos in other_positions[nr_deletions:]:
        new_char = get_random_char()
        while new_char == perturbed_sentence[2 * pos]:
            new_char = get_random_char()
        perturbed_sentence = perturbed_sentence[:2 * pos] + new_char + perturbed_sentence[2 * pos + 1:]

    return ''.join(char if char != custom_char else '' for char in perturbed_sentence)


class RandCharV5Trainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alphabet = alphabet
        self.q = args.q

        self.base_path = f'rand_char_v5_q{self.q}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with gradient-based character perturbations.
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

                # Tokenize inputs and get offsets
                tokenized_inputs = self.model.tokenize(inputs, return_offsets_mapping=True)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)
                offsets = tokenized_inputs['offset_mapping']

                self.optimizer.zero_grad()
                embeddings = self.model.input_embeddings(input_ids, attention_mask)
                outputs = self.model.forward_embeddings(embeddings, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)
                loss.backward(retain_graph=True)

                # Compute gradients for token embeddings
                token_gradients = embeddings.grad

                # Map gradients to character-level distributions
                char_distributions = [
                    map_gradients_to_characters(sentence, token_gradients[i], offsets[i])
                    for i, sentence in enumerate(inputs)
                ]

                # Perturb sentences
                perturbed_inputs = [
                    perturb_sentence_with_distribution(sentence, char_distributions[i], self.alphabet, q=self.q)
                    for i, sentence in enumerate(inputs)
                ]

                # Tokenize perturbed inputs
                tokenized_perturbed_inputs = self.model.tokenize(perturbed_inputs)
                input_ids_perturbed = tokenized_perturbed_inputs['input_ids'].to(self.device)
                attention_mask_perturbed = tokenized_perturbed_inputs['attention_mask'].to(self.device)
                
                # Forward pass with perturbed inputs
                outputs_perturbed = self.model.forward(input_ids=input_ids_perturbed,
                                                       attention_mask=attention_mask_perturbed)
                loss_perturbed = self.loss_fn(outputs_perturbed.logits, labels)

                # Combine losses and optimize
                total_loss = (loss + loss_perturbed) / 2
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                scheduler.step()

                # Log batch loss
                pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})

            # Epoch logging
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
