from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import os

def adversarial_perturbation(embeddings, model, attention_mask, labels, alpha, epsilon=0.3, attack_iters=3):
    # Get the gradients of the model
    perturbed_embeddings = embeddings.detach()
    perturbed_embeddings.requires_grad = True
    for _ in range(attack_iters):
        model.zero_grad()
        outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
        loss = CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        
        # Get the sign of the gradients
        gradients = perturbed_embeddings.grad.sign()
        
        # Perturb the embeddings in-place
        perturbed_embeddings = perturbed_embeddings + alpha * gradients

        # make sure that the embeddings are in the epsilon ball
        # perturbed_embeddings = torch.clamp(perturbed_embeddings, embeddings-epsilon, embeddings+epsilon)
        
        perturbed_embeddings = perturbed_embeddings.detach()
        perturbed_embeddings.requires_grad = True

    return perturbed_embeddings

class AdversarialTrainer:
    def __init__(self, model, tokenizer, optimizer, loss, device):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def train(self, 
              train_loader,
              val_loader,
              start_epoch=0,
              num_epochs=3,
              alpha=1e-3,
              beta=0.5,
              attack_iters=3,
              save_dir='saved_models'):
        self.model.train()

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        print('Started Training')

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get the token embeddings from the input_ids
                embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
                perturbed_embeddings = adversarial_perturbation(embeddings, self.model, attention_mask, labels, alpha, attack_iters=3)

                # Forward pass with perturbed embeddings
                self.model.zero_grad()
                outputs_adv = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = beta * self.loss(outputs.logits, labels) + (1-beta) * self.loss(outputs_adv.logits, labels)

                pbar.set_postfix({'Loss': f'{loss.item():.4}'})
                total_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

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

                # Calculate and display batch accuracy
                batch_acc = (predictions == labels).sum().item() / labels.size(0)
                pbar.set_postfix({'Acc': f'{batch_acc:.4f}'})

        accuracy = total_acc / total_count
        print(f"Validation Accuracy: {accuracy:.4f}")
