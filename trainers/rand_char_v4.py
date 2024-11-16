import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper
import random

def perturb_sentence(sentence, alphabet, alphabet_distribution, position_distribution, q=5, insertion_rate=0.5, deletion_rate=0.25, custom_char="§"):
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N / 100)
    nr_insertions = int(nr_perturbations * insertion_rate)
    nr_deletions = int(nr_perturbations * deletion_rate)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        # Sample from the alphabet distribution
        return random.choice(alphabet, weights=alphabet_distribution, k=1)[0]

    # Perform insertions
    insert_positions = random.choices(range(N), weights=position_distribution, k=nr_insertions)
    for pos in insert_positions:
        new_char = get_random_char()
        modified_pos = 2 * pos + 1
        perturbed_sentence = perturbed_sentence[:modified_pos] + new_char + perturbed_sentence[modified_pos + 1:]

    other_positions = random.choices(range(N), weights=position_distribution, k=nr_deletions + nr_swaps)

    # Perform deletions
    delete_positions = other_positions[:nr_deletions]
    for pos in delete_positions:
        modified_pos = 2 * pos
        perturbed_sentence = perturbed_sentence[:modified_pos] + custom_char + perturbed_sentence[modified_pos + 1:]

    # Perform swaps
    swap_positions = other_positions[nr_deletions:]
    for pos in swap_positions:
        new_char = get_random_char()
        while new_char == perturbed_sentence[2 * pos]:
            new_char = get_random_char()
        modified_pos = 2 * pos
        perturbed_sentence = perturbed_sentence[:modified_pos] + new_char + perturbed_sentence[modified_pos + 1:]
    
    return ''.join(char if char != custom_char else '' for char in perturbed_sentence)


class RandCharV4Trainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alphabet = alphabet

        self.q = args.q
        self.insertion_rate = args.insertion_rate
        self.deletion_rate = args.deletion_rate
        self.alph_dist = args.alph_dist
        self.pos_dist = args.pos_dist
        
        self.base_path = f'rand_char_v4_q{self.q}_ir{self.insertion_rate}_dr{self.deletion_rate}_ad{args.alph_dist}_pd{args.pos_dist}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--insertion_rate', type=float, default=0.33, help='Insertion rate')
        parser.add_argument('--deletion_rate', type=float, default=0.33, help='Deletion rate')
        parser.add_argument('--alph_dist', type=str, default='uniform', help='Alphabet sampling distribution type. How to sample characters for perturbations.')
        parser.add_argument('--pos_dist', type=str, default='uniform', help='Position sampling distribution type. How to sample positions for perturbations.')

        return parser
    
    def compute_alphabet_distribution(self):
        if self.alph_dist == 'uniform':
            return [1 / len(self.alphabet)] * len(self.alphabet)
    
    def compute_position_distribution(self, input_sentence):
        if self.pos_dist == 'uniform':
            return [1 / len(input_sentence)] * len(input_sentence)

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with random character perturbations.
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
                # Perturb inputs
                inputs = [perturb_sentence(sentence, self.alphabet, self.compute_alphabet_distribution(), self.compute_position_distribution(sentence), q=self.q) for sentence in batch['sentence']]
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
