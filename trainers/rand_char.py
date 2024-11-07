import os
import random
import string
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from transformers import AdamW

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

class RandCharTrainer:
    def __init__(self, model_wrapper, alphabet, device, args):
        self.model = model_wrapper
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = CrossEntropyLoss()
        self.device = device
        self.q = args.q
        self.alphabet = alphabet

        # Save path setup for organized experiment logging
        self.base_path = f'rand_char_q{str(self.q).replace(".", "_")}'
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs=3):
        """
        Main training loop with random character perturbations.
        """
        self.model.train()
        os.makedirs(save_path, exist_ok=True)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.03 * total_steps)
        warmup_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                inputs = [perturb_sentence(sentence, self.alphabet, q=self.q) for sentence in batch['sentence']]
                labels = batch['label'].to(self.device)

                # Tokenize perturbed inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)
                pbar.set_postfix({'Loss': f'{loss.item():.4}'})
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_grad_norm=0.3)
                self.optimizer.step()

                # Scheduler step
                if epoch * len(train_loader) + batch_idx < warmup_steps:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            # Validation and model saving
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            self.model.train()

            # Save model and tokenizer
            epoch_save_path = os.path.join(f'{save_path}_{self.base_path}', f'model_e{epoch+1}')
            self.model.save(epoch_save_path)
            print(f"Model saved to {epoch_save_path}")