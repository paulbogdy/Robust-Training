import random
import string
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
from tqdm import tqdm
import os
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

def perturb_sentence(sentence, alphabet, q=0.05, custom_char="§"):
    N = len(sentence)
    perturbed_sentence = ''.join([char + custom_char for char in sentence])

    nr_perturbations = int(q * N)
    nr_insertions = int(nr_perturbations / 3)
    nr_deletions = int(nr_perturbations / 3)
    nr_swaps = nr_perturbations - nr_insertions - nr_deletions

    def get_random_char():
        return random.choice(alphabet)

    # Perform insertions
    insert_positions = random.sample(range(N), nr_insertions)
    for pos in insert_positions:
        perturbed_sentence[2*pos + 1] = get_random_char()
    
    # Perform deletions
    delete_positions = random.sample(range(N), nr_deletions)
    for pos in delete_positions:
        perturbed_sentence[2*pos] = custom_char

    # Perform swaps
    swap_positions = random.sample(range(N), nr_swaps)
    for pos in swap_positions:
        new_char = get_random_char()
        while (new_char == perturbed_sentence[2*pos]):
            new_char = get_random_char()
        perturbed_sentence[2*pos] = new_char
    
    return ''.join(char if char != custom_char else '' for char in perturbed_sentence)

class PerturbedTrainer:
    def __init__(self, model, tokenizer, optimizer, loss, device, alphabet=string.ascii_lowercase):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.alphabet = alphabet

    def train(self, train_loader, val_loader, q=0.05, num_epochs=3, max_grad_norm=0.3, save_dir='saved_models'):
        os.makedirs(save_dir, exist_ok=True)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.03 * total_steps)
        
        warmup_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))
            
            for batch_idx, batch in pbar:
                # Apply perturbations to each sentence in the batch
                inputs = [perturb_sentence(sentence, self.alphabet, q=q) for sentence in batch['input_sentence']]
                labels = batch['label'].to(self.device)

                # Tokenize perturbed inputs
                tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss(outputs.logits, labels)
                pbar.set_postfix({'Loss': f'{loss.item():.4}'})
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                # Scheduler step
                if epoch * len(train_loader) + batch_idx < warmup_steps:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
            
            self.validate(val_loader)
            self.model.train()
            
            # Save the model
            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}')
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    def validate(self, val_loader):
        self.model.eval()
        total_acc = 0
        total_count = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                total_acc += (predictions == labels).sum().item()
                total_count += labels.size(0)

                batch_acc = (predictions == labels).sum().item() / labels.size(0)
                pbar.set_postfix({'Acc': f'{batch_acc:.4f}'})

        accuracy = total_acc / total_count
        print(f"Validation Accuracy: {accuracy:.4f}")