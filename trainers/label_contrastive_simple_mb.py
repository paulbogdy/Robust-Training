import torch
import os
from tqdm import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
from models.model_wrapper import ModelWrapper
import torch.nn.functional as F
import random

class ClassWiseMemoryBank:
    def __init__(self, size, feature_dim, num_classes, device):
        self.size = size  # Maximum samples per class
        self.device = device
        self.num_classes = num_classes

        # Create a dictionary of class-wise queues
        self.memory = {c: torch.zeros(size, feature_dim).to(device) for c in range(num_classes)}
        self.ptr = {c: 0 for c in range(num_classes)}  # Track insert position for each class

    def update(self, features, labels):
        """
        Update memory bank for each class.
        """
        for i in range(features.shape[0]):  # Iterate through batch samples
            c = labels[i].item()  # Get class label
            self.memory[c][self.ptr[c] % self.size] = features[i].detach()  # Store new feature
            self.ptr[c] = (self.ptr[c] + 1) % self.size  # FIFO update

    def get_negatives(self, label):
        """
        Get negative samples for a given label.
        """
        negatives = torch.cat([self.memory[c] for c in range(self.num_classes) if c != label], dim=0)
        return negatives
    
class LabelContrastiveSimpleMBTrainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1

        self.alphabet = alphabet

        self.q = args.q  # Perturbation rate
        self.alpha = args.alpha  # Weight for contrastive loss
        self.temperature = args.temperature  # Temperature parameter for contrastive loss
        self.bank_size = args.bank_size  # Size of the memory bank per class

        # Get hidden size from model
        hidden_size = self.model.model.config.hidden_size
        projection_dim = args.projection_dim

        self.bank = ClassWiseMemoryBank(self.bank_size, projection_dim, self.model.num_labels(), self.device)

        # Define projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        ).to(self.device)

        # Define optimizer including parameters of the model and projection head
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr=args.learning_rate, weight_decay=0.01
        )

        # Define loss functions
        self.loss_fn = CrossEntropyLoss()

        self.base_path = f'label_contrastive_simple_mb_q{self.q}_a{str(self.alpha).replace(".","_")}_t{str(self.temperature).replace(".", "_")}_p{projection_dim}_bank{self.bank_size}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--alpha', type=float, default=0.5, help='Weight for contrastive loss')
        parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')
        parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension for contrastive loss')
        parser.add_argument('--bank_size', type=int, default=256, help='Size of the memory bank pe class, the nr of negatives is (num_classes - 1) * bank_size')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with contrastive learning.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(self.warmup_proportion * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            self.projection_head.train()
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

            for batch_idx, batch in pbar:
                # Original (unperturbed) inputs
                unperturbed_inputs = batch['sentence']
                labels = batch['label'].to(self.device)

                # Tokenize both perturbed and unperturbed inputs
                unperturbed_tokenized = self.model.tokenize(unperturbed_inputs)

                input_ids_unperturbed = unperturbed_tokenized['input_ids'].to(self.device)
                attention_mask_unperturbed = unperturbed_tokenized['attention_mask'].to(self.device)

                # Forward pass for unperturbed inputs (for main classification loss)
                self.optimizer.zero_grad()
                outputs_unperturbed = self.model.forward(
                    input_ids=input_ids_unperturbed,
                    attention_mask=attention_mask_unperturbed,
                    output_hidden_states=True
                )
                unperturbed_logits = outputs_unperturbed.logits

                # Calculate cross-entropy loss for unperturbed sample
                cross_entropy_loss = self.loss_fn(unperturbed_logits, labels)

                # Extract [CLS] embeddings from the last hidden state of both perturbed and unperturbed inputs
                cls_embedding_unperturbed = outputs_unperturbed.hidden_states[-1][:, 0, :]  # [CLS] embedding for unperturbed

                # Pass [CLS] embeddings through the projection head
                z_i = self.projection_head(cls_embedding_unperturbed)

                z_i = F.normalize(z_i, dim=1)

                self.bank.update(z_i, labels)

                contrastive_loss = self.contrastive_loss_fn(z_i, labels)

                # Total loss
                total_loss_value = (1 - self.alpha) * cross_entropy_loss + self.alpha * contrastive_loss

                # Backward pass and optimization
                total_loss_value.backward()
                clip_grad_norm_(list(self.model.parameters()) + list(self.projection_head.parameters()), self.max_grad_norm)
                self.optimizer.step()
                scheduler.step()

                # Log batch loss
                total_loss += total_loss_value.item()
                pbar.set_postfix({'Loss': f'{total_loss_value.item():.4f}', 'CE Loss': f'{cross_entropy_loss.item():.4f}', 'Contrastive Loss': f'{contrastive_loss.item():.4f}'})

            # Epoch logging
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = self.model.evaluate(val_loader, self.device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        torch.save(self.projection_head.state_dict(), os.path.join(model_path, 'projection_head.pt'))
        print(f"Model and projection head saved to {model_path}")

    def contrastive_loss_fn(self, embeddings, labels):
        """
        Compute the InfoNCE loss for contrastive learning given two sets of embeddings.

        Args:
            z_i (Tensor): Embeddings from the first set (batch_size, projection_dim).
            z_j (Tensor): Embeddings from the second set (batch_size, projection_dim).
            temperature (float): Temperature parameter for scaling.

        Returns:
            Tensor: The computed InfoNCE loss.
        """
        loss = None
        unique_labels, counts = torch.unique(labels, return_counts=True)

        nr_classes = 0
        for label, num in zip(unique_labels, counts):
            if num.item() < 2:
                continue
            nr_classes += 1
            queries = embeddings[labels == label]
            keys = self.bank.get_negatives(label)

            positives = torch.matmul(queries, queries.T) 
            mask = torch.eye(positives.shape[0], dtype=torch.bool).to(self.device)
            positives = positives[~mask].view(positives.shape[0], -1)
            negatives = torch.matmul(queries, keys.T)

            logits = torch.cat([positives, negatives], dim=1)
            positions = torch.arange(positives.shape[0], dtype=torch.long).to(self.device)

            logits /= self.temperature

            if loss is None:
                loss = F.cross_entropy(logits, positions) / positives.shape[0]
            else:
                loss += F.cross_entropy(logits, positions) / positives.shape[0]

        return loss / nr_classes
