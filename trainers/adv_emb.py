import os
from pathlib import Path
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, AdamW
from models.model_wrapper import ModelWrapper
from torch.nn.utils import clip_grad_norm_

class AdvEmbTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        self.loss = CrossEntropyLoss()
        self.device = device

        self.alpha = args.alpha
        self.beta = args.beta
        self.attack_iters = args.attack_iters

        self.base_path = f'adv_emb_a{str(self.alpha).replace(".", "_")}_b{str(self.beta).replace(".", "_")}_attack_i{self.attack_iters}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--alpha', type=float, default=1e-3)
        parser.add_argument('--beta', type=float, default=0.5)
        parser.add_argument('--attack_iters', type=int, default=5)
        return parser

    def adversarial_perturbation(self, embeddings, attention_mask, labels, alpha, attack_iters):
        # Get the gradients of the model
        perturbed_embeddings = embeddings.detach()
        perturbed_embeddings.requires_grad = True
        for _ in range(attack_iters):
            self.model.zero_grad()
            outputs = self.model.forward_embeddings(input_embeds=perturbed_embeddings, attention_mask=attention_mask)
            loss = self.loss(outputs.logits, labels)
            loss.backward()
            
            # Get the sign of the gradients
            gradients = perturbed_embeddings.grad.sign()
            
            # Perturb the embeddings in-place
            perturbed_embeddings = perturbed_embeddings + alpha * gradients
            
            perturbed_embeddings = perturbed_embeddings.detach()
            perturbed_embeddings.requires_grad = True

        return perturbed_embeddings

    def train(self, 
              train_loader,
              val_loader,
              save_path,
              num_epochs=5):
        self.model.train()

        max_grad_norm = 0.3
        total_steps = len(train_loader) * num_epochs

        warmup_steps = int(0.03 * total_steps)
        
        warmup_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))
            for batch_idx, batch in pbar:
                inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                # Tokenize inputs
                tokenized_inputs = self.model.tokenize(inputs)
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)

                # Forward pass with perturbed embeddings
                self.model.zero_grad()
                if (self.beta == 0):
                    # Get the token embeddings from the input_ids
                    embeddings = self.model.input_embeddings(input_ids)
                    perturbed_embeddings = self.adversarial_perturbation(embeddings, attention_mask, labels, self.alpha, self.attack_iters)

                    outputs_adv = self.model.forward_embeddings(input_embeds=perturbed_embeddings, attention_mask=attention_mask)
                    loss = self.loss(outputs_adv.logits, labels)
                elif (self.beta == 1):
                    outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.loss(outputs.logits, labels)
                else:
                    # Get the token embeddings from the input_ids
                    embeddings = self.model.input_embeddings(input_ids)
                    perturbed_embeddings = self.adversarial_perturbation(embeddings, attention_mask, labels, self.alpha, self.attack_iters)

                    outputs_adv = self.model.forward_embeddings(input_embeds=perturbed_embeddings, attention_mask=attention_mask)
                    outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.beta * self.loss(outputs.logits, labels) + (1-self.beta) * self.loss(outputs_adv.logits, labels)

                pbar.set_postfix({'Loss': f'{loss.item():.4}'})
                total_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                if epoch * len(train_loader) + batch_idx < warmup_steps:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            # Validate the model
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            self.model.train()
            
            # Save the model
            save_path = os.path.join(f'{save_path}_{self.base_path}', f'model_e{epoch+1}')
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
