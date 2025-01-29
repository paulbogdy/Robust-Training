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

class LabelContrastiveTrainer:
    def __init__(self, model_wrapper: ModelWrapper, alphabet, device, args):
        self.model = model_wrapper
        self.device = device
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1

        self.alphabet = alphabet

        self.q = args.q  # Perturbation rate
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

        self.base_path = f'label_contrastive_q{self.q}_t{str(self.temperature).replace(".", "_")}_p{projection_dim}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--q', type=int, default=5, help='Perturbation rate %(0-100)')
        parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')
        parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension for contrastive loss')
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
                outputs_unperturbed = self.model.forward(
                    input_ids=input_ids_unperturbed,
                    attention_mask=attention_mask_unperturbed,
                    output_hidden_states=True
                )
                unperturbed_logits = outputs_unperturbed.logits

                # Forward pass for perturbed inputs to obtain [CLS] embeddings
                outputs_perturbed = self.model.forward(
                    input_ids=input_ids_perturbed,
                    attention_mask=attention_mask_perturbed,
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

                contrastive_loss = self.contrastive_loss_fn(z_i, z_j, labels)

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

    def contrastive_loss_fn(self, z_i, z_j, labels):
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
        extended_labels = labels.repeat(2)  # Shape: (2N,)

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, dim=1)

         # Compute the similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (2N, 2N)

        # Mask to remove self-similarities
        mask = torch.eye(extended_labels.shape[0], dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Compute unique labels
        unique_labels, counts = torch.unique(extended_labels, return_counts=True)

        loss = None

        for label in unique_labels:
            if counts[label] >= 2:
                positive_rows = extended_labels == label
                positive_mat = (positive_rows.unsqueeze(0) & positive_rows.unsqueeze(1)).float()
                positive_mat = positive_mat[~mask].view(positive_mat.shape[0], -1)
                positive_mat = positive_mat[positive_rows]

                similarity_mat = similarity_matrix[positive_rows]

                positive_sim = similarity_mat[positive_mat.bool()].view(positive_mat.shape[0], -1)
                negative_sim = similarity_mat[~positive_mat.bool()].view(positive_mat.shape[0], -1)

                logits = torch.cat([positive_sim, negative_sim], dim=1)
                positions = torch.arange(logits.shape[0], dtype=torch.long).to(self.device)
                positions[positions >= positive_sim.shape[0]] = -1

                logits /= self.temperature

                if loss is None:
                    loss = F.cross_entropy(logits, positions, ignore_index=-1)
                else:
                    loss += F.cross_entropy(logits, positions, ignore_index=-1)

        return loss
