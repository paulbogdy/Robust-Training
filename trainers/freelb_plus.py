import os
from pathlib import Path
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from models.model_wrapper import ModelWrapper
from torch.nn.utils import clip_grad_norm_
import torch

class FreeLBPlusTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        self.loss_fn = CrossEntropyLoss()
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0

        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.k = args.k

        self.base_path = f'freelb_plus_a{str(self.alpha).replace(".", "_")}_e{str(self.epsilon).replace(".", "_")}_k{self.k}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--epsilon', type=float, default=0.6)
        parser.add_argument('--k', type=int, default=10)
        return parser

    def train(self, 
              train_loader,
              val_loader,
              save_path,
              num_epochs=5):
        
        model_path = f'{save_path}_{self.base_path}'
        os.makedirs(model_path, exist_ok=True)

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

                self.optimizer.zero_grad()
                self.model.zero_grad()

                embeddings = self.model.input_embeddings(input_ids)
                batch_size, seq_len, emb_dim = embeddings.size()
                delta = torch.rand_like(embeddings).uniform_(-self.epsilon, self.epsilon) / torch.sqrt(torch.tensor(seq_len * emb_dim, dtype=torch.float32))
                
                acc_loss = 0
                attack_iters = self.k
                if warmup_steps < epoch * len(train_loader) + batch_idx:
                    attack_iters = 2

                for t in range(attack_iters):
                    delta.requires_grad_()
                    outputs = self.model.forward_embeddings(input_embeds=embeddings + delta, attention_mask=attention_mask)

                    loss = self.loss_fn(outputs.logits, labels)/attack_iters
                    acc_loss += loss.item()
                    loss.backward()

                    if t == attack_iters - 1:
                        break

                    delta_grad = delta.grad.clone().detach()

                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + self.alpha * delta_grad / denorm).detach()

                    embeddings = self.model.input_embeddings(input_ids)

                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                scheduler.step()

                # Logging loss
                total_loss += acc_loss
                pbar.set_postfix({'Loss': f'{acc_loss:.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

            # Validate the model
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
