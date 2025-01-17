import os
from pathlib import Path
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, AdamW
from models.model_wrapper import ModelWrapper
from torch.nn.utils import clip_grad_norm_
import torch

class FgsmTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.epsilon = args.epsilon

        self.base_path = f'fgsm_e{str(self.epsilon).replace(".", "_")}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--epsilon', type=float, default=1e-3)
        return parser

    def train(self, 
              train_loader,
              val_loader,
              save_path,
              num_epochs=5):
        
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        # Total steps for learning rate scheduling
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        # Learning rate schedulers
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))
            for batch_idx, batch in pbar:
                inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                # Tokenize inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Step 1: Get input embeddings
                embeddings = self.model.input_embeddings(input_ids).detach()
                embeddings.requires_grad = True

                # Step 2: Forward pass for clean embeddings (Optional)
                outputs = self.model.forward_embeddings(input_embeds=embeddings, attention_mask=attention_mask)
                clean_loss = self.loss_fn(outputs.logits, labels)

                # Step 3: Compute gradients for FGSM
                clean_loss.backward()
                gradients = embeddings.grad.data

                # Step 4: Generate FGSM perturbation
                delta = self.epsilon * torch.sign(gradients)

                # Step 5: Forward pass with perturbed embeddings
                embeddings.grad.data.zero_()  # Clear gradients for the next backward pass
                outputs_pert = self.model.forward_embeddings(input_embeds=embeddings + delta, attention_mask=attention_mask)
                loss = self.loss_fn(outputs_pert.logits, labels)

                # Step 6: Backward pass for perturbed embeddings
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Step 7: Optimization step
                self.optimizer.step()

                # Update scheduler (if needed)
                scheduler.step()

                # Logging loss
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

            # Validate the model
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
