import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper
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


import torch
import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_

class RandCharV2Trainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.smooth_loss_fn = MSELoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alphabet = alphabet

        self.q = args.q
        self.alpha = args.alpha  # Weight for smooth loss
        
        self.base_path = f'rand_char_v2_q{self.q}_a{str(self.alpha).replace(".", "_")}'
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--alpha', type=float, default=1, help='Weight for smooth loss')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with random character perturbations.
        """
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                # Original (unperturbed) inputs
                unperturbed_inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                # Perturb inputs
                perturbed_inputs = [perturb_sentence(sentence, self.alphabet, q=self.q) for sentence in unperturbed_inputs]

                # Tokenize both perturbed and unperturbed inputs
                unperturbed_tokenized = self.model.tokenize(unperturbed_inputs)
                perturbed_tokenized = self.model.tokenize(perturbed_inputs)

                input_ids_unperturbed = unperturbed_tokenized['input_ids'].to(self.device)
                attention_mask_unperturbed = unperturbed_tokenized['attention_mask'].to(self.device)
                
                input_ids_perturbed = perturbed_tokenized['input_ids'].to(self.device)
                attention_mask_perturbed = perturbed_tokenized['attention_mask'].to(self.device)

                # Forward pass for unperturbed inputs (for main classification loss)
                self.optimizer.zero_grad()
                outputs_unperturbed = self.model.forward(input_ids=input_ids_unperturbed, attention_mask=attention_mask_unperturbed, output_hidden_states=True)
                unperturbed_logits = outputs_unperturbed.logits

                # Calculate cross-entropy loss for unperturbed sample
                cross_entropy_loss = self.loss_fn(unperturbed_logits, labels)

                # Forward pass for perturbed inputs to obtain [CLS] embeddings
                outputs_perturbed = self.model.forward(input_ids=input_ids_perturbed, attention_mask=attention_mask_perturbed, output_hidden_states=True)
                
                # Extract [CLS] embeddings from the last hidden state of both perturbed and unperturbed inputs
                cls_embedding_unperturbed = outputs_unperturbed.hidden_states[-1][:, 0, :]  # [CLS] embedding for unperturbed
                cls_embedding_perturbed = outputs_perturbed.hidden_states[-1][:, 0, :]      # [CLS] embedding for perturbed

                # Compute smooth loss (L2 loss) between [CLS] embeddings of unperturbed and perturbed
                smooth_loss = self.smooth_loss_fn(cls_embedding_unperturbed, cls_embedding_perturbed)

                # Combine cross-entropy loss and smooth loss, weighted by alpha
                total_loss_value = cross_entropy_loss + self.alpha * smooth_loss

                # Backward pass and optimization
                total_loss_value.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                scheduler.step()

                # Log batch loss
                total_loss += total_loss_value.item()
                pbar.set_postfix({'Loss': f'{total_loss_value.item():.4f}'})

            # Epoch logging
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            # Save model
            epoch_save_path = os.path.join(f'{save_path}_{self.base_path}', f'model_e{epoch+1}')
            self.model.save(epoch_save_path)
            print(f"Model saved to {epoch_save_path}")