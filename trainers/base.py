import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper

class BaseTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        """
        Initialize the BaseTrainer with model, device, optimizer, and other configurations.
        """
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.base_path = 'base'

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop.
        """
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

                # Tokenize inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)

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
            self.model.train()

            # Save the model after each epoch
            epoch_save_path = os.path.join(f'{save_path}_{self.base_path}', f'model_e{epoch+1}')
            self.model.save(epoch_save_path)
            print(f"Model saved to {epoch_save_path}")
