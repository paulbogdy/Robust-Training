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

class ContrastiveV4Trainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.device = device
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1

        self.epsilon = args.epsilon  # Perturbation magnitude
        self.alpha = args.alpha  # Weight for contrastive loss
        self.temperature = args.temperature  # Temperature parameter for contrastive loss

        # Get hidden size from model
        hidden_size = self.model.model.config.hidden_size
        projection_dim = args.projection_dim

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

        self.base_path = f'contrastive_v4_e{str(self.epsilon).replace(".", "_")}_a{str(self.alpha).replace(".", "_")}_t{str(self.temperature).replace(".", "_")}_p{projection_dim}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--epsilon', type=float, default=1e-3, help='Perturbation magnitude in L-infinity norm')
        parser.add_argument('--alpha', type=float, default=0.5, help='Weight for contrastive loss')
        parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')
        parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension for contrastive loss')
        return parser

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop with contrastive learning.
        """
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
                loss = self.loss_fn(unperturbed_logits, labels)

                loss.backward(retain_graph=True)
                embeddings_matrix_grad = self.model.model.base_model.embeddings.word_embeddings.weight.grad

                # Adversarial perturbation
                perturbation = perturbation = self.epsilon * embeddings_matrix_grad.sign()

                with torch.no_grad():
                    perturbed_embeddings = (
                        self.model.model.base_model.embeddings.word_embeddings.weight + perturbation
                    )

                # Forward pass for perturbed inputs to obtain [CLS] embeddings
                outputs_perturbed = self.model.forward(
                    input_ids=perturbed_embeddings[input_ids_unperturbed],
                    attention_mask=attention_mask_unperturbed,
                    output_hidden_states=True
                )
                perturbed_logits = outputs_perturbed.logits
                
                # Calculate cross-entropy loss for unperturbed sample
                cross_entropy_loss = (self.loss_fn(unperturbed_logits, labels) + self.loss_fn(perturbed_logits, labels)) / 2

                # Extract [CLS] embeddings from the last hidden state of both perturbed and unperturbed inputs
                cls_embedding_unperturbed = outputs_unperturbed.hidden_states[-1][:, 0, :]  # [CLS] embedding for unperturbed
                cls_embedding_perturbed = outputs_perturbed.hidden_states[-1][:, 0, :]      # [CLS] embedding for perturbed

                # Pass [CLS] embeddings through the projection head
                z_i = self.projection_head(cls_embedding_unperturbed)
                z_j = self.projection_head(cls_embedding_perturbed)

                contrastive_loss = self.contrastive_loss_fn(z_i, z_j)

                # Total loss
                total_loss_value = (1 - self.alpha) * cross_entropy_loss + self.alpha * contrastive_loss

                # Backward pass and optimization
                total_loss_value.backward()
                clip_grad_norm_(list(self.model.parameters()) + list(self.projection_head.parameters()), self.max_grad_norm)
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

            # Save model and projection head
            epoch_save_path = os.path.join(f'{save_path}_{self.base_path}', f'model_e{epoch+1}')
            if not os.path.exists(epoch_save_path):
                os.makedirs(epoch_save_path)
            self.model.save(epoch_save_path)
            torch.save(self.projection_head.state_dict(), os.path.join(epoch_save_path, 'projection_head.pt'))
            print(f"Model and projection head saved to {epoch_save_path}")

    def contrastive_loss_fn(self, z_i, z_j):
        """
        Compute the InfoNCE loss for contrastive learning given two sets of embeddings.

        Args:
            z_i (Tensor): Embeddings from the first set (batch_size, projection_dim).
            z_j (Tensor): Embeddings from the second set (batch_size, projection_dim).
            temperature (float): Temperature parameter for scaling.

        Returns:
            Tensor: The computed InfoNCE loss.
        """
        # Combine the embeddings
        embeddings = torch.cat([z_i, z_j], dim=0)  # Shape: (2N, projection_dim)

         # Compute the similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (2N, 2N)

        # Create labels
        batch_size = z_i.size(0)
        labels = torch.arange(batch_size).repeat(2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Mask to remove self-similarities
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits /= self.temperature
        return F.cross_entropy(logits, labels)
