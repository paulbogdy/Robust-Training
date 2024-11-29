import os
from pathlib import Path
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from models.model_wrapper import ModelWrapper
import torch

class FreeLBTrainer:
    def __init__(self, model_wrapper: ModelWrapper, device, args):
        self.model = model_wrapper
        self.device = device
        self.learning_rate = args.learning_rate
        self.loss_fn = CrossEntropyLoss()

        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.k = args.k

        self.base_path = f'freelb_a{str(self.alpha).replace(".", "_")}_e{str(self.epsilon).replace(".", "_")}_k{self.k}'

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

                embeddings = self.model.input_embeddings(input_ids)
                batch_size, seq_len, emb_dim = embeddings.size()
                perturbation = torch.rand_like(embeddings).uniform_(-self.epsilon, self.epsilon) / torch.sqrt(seq_len * emb_dim)
                perturbation.requires_grad = True

                acc_grad = None
                acc_loss = 0
                for t in range(self.k):
                    perturbed_input = embeddings + perturbation
                    self.model.model.zero_grad()
                    outputs = self.model.forward_embeddings(input_embeds=perturbed_input, attention_mask=attention_mask)
                    loss = self.loss_fn(outputs.logits, labels)

                    loss.backward()
                    acc_loss += loss.item()
                    if acc_grad is None:
                        acc_grad = [
                            (param.grad.clone() / self.k if param.grad is not None else torch.zeros_like(param))
                            for param in self.model.model.parameters()
                        ]
                    else:
                        for i, param in enumerate(self.model.model.parameters()):
                            if param.grad is not None:
                                acc_grad[i] += param.grad.clone() / self.k


                    grad_adv = perturbation.grad.detach()
                    frob_norm = torch.norm(grad_adv.view(batch_size, -1), dim=1).view(batch_size, 1, 1) + 1e-8
                    perturbation = (perturbation + self.alpha * grad_adv/frob_norm).detach()
                    perturbation.requires_grad = True

                for i, param in enumerate(self.model.model.parameters()):
                    if param.grad is not None:
                        param.data -= self.learning_rate * acc_grad[i]

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
