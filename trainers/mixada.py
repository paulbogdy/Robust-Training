import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from models.model_wrapper import ModelWrapper
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
import pandas as pd

class MixAdaTrainer:
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

        self.aug_folder = args.aug_folder
        self.aug_type = args.aug_type

        self.alpha = args.alpha

        self.base_path = f'mixada_{args.aug_type}'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--aug_folder', type=str, default=None, help='Path to the folder with augmented data')
        parser.add_argument('--aug_type', type=str, default='charmer', help='Type of augmentation to use', choices=['charmer', 'all'])
        parser.add_argument('--alpha', type=float, default=2, help='The param for the beta distribution')
        return parser

    def augment_train_data(self, train_loader):
        """
        Augment the training data with the specified augmentation type.
        """
        # Load the original dataset from the datasets library
        original_sentences = train_loader.dataset["sentence"]
        original_labels = train_loader.dataset["label"]

        # Load augmentation files
        augmentation_files = []
        if self.aug_type == "charmer":
            # Filter for charmer_10.csv only
            charmer_file = os.path.join(self.aug_folder, "charmer_10.csv")
            if os.path.exists(charmer_file):
                augmentation_files.append(charmer_file)
        elif self.aug_type == "all":
            # Include all .csv files in the augmentation folder
            augmentation_files = [
                os.path.join(self.aug_folder, f)
                for f in os.listdir(self.aug_folder)
                if f.endswith(".csv")
            ]

        # Process augmentation files
        augmented_sentences = []
        augmented_labels = []

        for file in augmentation_files:
            df = pd.read_csv(file)

            # Filter for successful attacks and extract perturbed sentences and true labels
            for _, row in df.iterrows():
                if pd.notna(row['perturbed']):
                    augmented_sentences.append(row['perturbed'])
                    augmented_labels.append(row['True'])

        # Combine original and augmented data
        combined_sentences = original_sentences + augmented_sentences
        combined_labels = original_labels + augmented_labels

        # Convert to Hugging Face Dataset
        combined_dataset = HFDataset.from_dict({
            "sentence": combined_sentences,
            "label": combined_labels
        })

        # Create a new DataLoader
        augmented_loader = DataLoader(
            combined_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            collate_fn=train_loader.collate_fn
        )

        return augmented_loader

    def train(self, train_loader, val_loader, save_path, num_epochs):
        """
        Main training loop.
        """
        model_path = f'{save_path}_{self.base_path}'
        os.mkdir(model_path)

        train_loader = self.augment_train_data(train_loader)

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
                input_embeds = self.model.input_embeddings(input_ids)
                batch_perm = torch.randperm(input_embeds.size(0))

                shuffled_input_embeds = input_embeds[batch_perm]

                soft_original_labels = torch.nn.functional.one_hot(labels, num_classes=self.model.num_labels()).float()
                soft_shuffled_labels = soft_original_labels[batch_perm]

                lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().to(self.device)

                # interpolate the [CLS] token embeddings
                mixed_input_embeds = lam * input_embeds + (1 - lam) * shuffled_input_embeds
                mixed_labels = lam * soft_original_labels + (1 - lam) * soft_shuffled_labels

                mixed_outputs = self.model.forward_embeddings(input_embeds=mixed_input_embeds, attention_mask=attention_mask)
                loss_mix = F.kl_div(F.log_softmax(mixed_outputs.logits, dim=-1), mixed_labels, reduction='batchmean')

                outputs = self.model.forward_embeddings(input_embeds=input_embeds, attention_mask=attention_mask)
                loss_ce = self.loss_fn(outputs.logits, labels)

                loss = loss_ce + loss_mix
                
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

        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
