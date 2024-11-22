import argparse
from models.model_wrapper import ModelWrapper
from data.dataloader import load_dataset
from utils import get_alphabet
from torch.utils.data import DataLoader
from trainers import *
import torch
import random
import numpy as np

def main(args):
     # Set the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset, num_labels = load_dataset(args.dataset_name)
    model_wrapper = ModelWrapper(args.model_name, num_labels=num_labels)

    # Set up data loaders based on preprocessing type
    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(dataset['validation'], batch_size=args.batch_size, shuffle=False)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper.model.to(device)

    # Select the training method and initialize accordingly
    if args.training_method == 'adv_emb':
        trainer = AdvEmbTrainer(model_wrapper, device, args)
    elif args.training_method == 'rand_char':
        trainer = RandCharTrainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_char_v2':
        trainer = RandCharV2Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_char_v3':
        trainer = RandCharV3Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_char_v4':
        trainer = RandCharV4Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_char_v5':
        trainer = RandCharV5Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_char_v6':
        trainer = RandCharV6Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive':
        trainer = ContrastiveTrainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive_v2':
        trainer = ContrastiveV2Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive_v3':
        trainer = ContrastiveV3Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive_v4':
        trainer = ContrastiveV4Trainer(model_wrapper, device, args)
    elif args.training_method == 'contrastive_v5':
        trainer = ContrastiveV5Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive_v6':
        trainer = ContrastiveV6Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'contrastive_v8':
        trainer = ContrastiveV8Trainer(model_wrapper, get_alphabet(args.dataset_name), device, args)
    elif args.training_method == 'rand_mask':
        trainer = RandMaskTrainer(model_wrapper, device, args)
    elif args.training_method == 'base':
        trainer = BaseTrainer(model_wrapper, device, args)

    save_path = f'{args.model_name}_{args.dataset_name}_seed{args.seed}_batch{args.batch_size}_lr{str(args.learning_rate).replace(".", "_")}'

    trainer.train(train_loader, val_loader, save_path=save_path, num_epochs=args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with adversarial training.")

    # General arguments
    parser.add_argument(
        '--model_name',
        type=str, 
        default='bert', 
        choices=['bert', 'roberta', 'albert'],
        help='Name of the model to train.')
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='sst', 
        choices=['sst'],
        help='Name of the dataset to train on.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.')
    parser.add_argument(
        '--training_method', 
        type=str, 
        choices=['adv_emb', 'rand_char', 'rand_char_v2', 'rand_char_v3', 'rand_char_v4', 'rand_char_v5', 'rand_char_v6', 'base', 'contrastive', 'contrastive_v2', 'contrastive_v3', 'contrastive_v4', 'contrastive_v5', 'contrastive_v6', 'contrastive_v8', 'rand_mask'], 
        required=True,
        help='Training method to use.')
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=5,
        help='Number of epochs to train.')
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training.')
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=2e-4,
        help='Learning rate for the optimizer.')

    # Add method-specific arguments dynamically
    args, remaining_args = parser.parse_known_args()
    if args.training_method == 'adv_emb':
        parser = AdvEmbTrainer.add_args(parser)
    elif args.training_method == 'rand_char':
        parser = RandCharTrainer.add_args(parser)
    elif args.training_method == 'rand_char_v2':
        parser = RandCharV2Trainer.add_args(parser)
    elif args.training_method == 'rand_char_v3':
        parser = RandCharV3Trainer.add_args(parser)
    elif args.training_method == 'rand_char_v4':
        parser = RandCharV4Trainer.add_args(parser)
    elif args.training_method == 'rand_char_v5':
        parser = RandCharV5Trainer.add_args(parser)
    elif args.training_method == 'rand_char_v6':
        parser = RandCharV6Trainer.add_args(parser)
    elif args.training_method == 'contrastive':
        parser = ContrastiveTrainer.add_args(parser)
    elif args.training_method == 'contrastive_v2':
        parser = ContrastiveV2Trainer.add_args(parser)
    elif args.training_method == 'contrastive_v3':
        parser = ContrastiveV3Trainer.add_args(parser)
    elif args.training_method == 'contrastive_v4':
        parser = ContrastiveV4Trainer.add_args(parser)
    elif args.training_method == 'contrastive_v5':
        parser = ContrastiveV5Trainer.add_args(parser)
    elif args.training_method == 'contrastive_v6':
        parser = ContrastiveV6Trainer.add_args(parser)
    elif args.training_method == 'contrastive_v8':
        parser = ContrastiveV8Trainer.add_args(parser)
    elif args.training_method == 'rand_mask':
        parser = RandMaskTrainer.add_args(parser)

    args = parser.parse_args()
    main(args)
