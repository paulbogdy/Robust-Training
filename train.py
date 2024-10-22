import argparse
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from models.model_factory import load_model_and_tokenizer
from data.dataloader import load_dataset
from data.preprocess import get_dataloader
from trainer.adversarial_embeddings import AdversarialTrainer

def main(args):
    model_name = args.model_name

    if (args.continue_training):
        model_name = f'./{args.save_dir}/model_epoch_{args.start_epoch}'

    print(f'Training adversarially for model: {model_name}')

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=args.num_labels)

    # Load dataset and dataloaders
    dataset = load_dataset(args.dataset_name)
    train_loader = get_dataloader(data=dataset['train'], tokenizer=tokenizer, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(data=dataset['validation'], tokenizer=tokenizer, batch_size=args.batch_size, shuffle=False)

    print(f'Loaded Train dataset with {len(train_loader.dataset)} samples')
    print(f'Loaded Validation dataset with {len(val_loader.dataset)} samples')

    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss = CrossEntropyLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    # Initialize the AdversarialTrainer
    adversarial_trainer = AdversarialTrainer(model=model, 
                                             tokenizer=tokenizer, 
                                             optimizer=optimizer, 
                                             loss=loss, 
                                             device=device)

    # Train the model
    adversarial_trainer.train(train_loader=train_loader,
                              val_loader=val_loader,
                              start_epoch=args.start_epoch,
                              num_epochs=args.num_epochs,
                              alpha=args.alpha,
                              beta=args.beta,
                              save_dir=args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with adversarial training.")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Name of the model to load.')
    parser.add_argument('--continue_training', type=bool, default=False, help='Whether or not to continue training.')
    parser.add_argument('--dataset_name', type=str, default='sst', help='Name of the dataset to load.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model.')
    parser.add_argument('--start_epoch', type=int, default=0, help='The starting epoch of the training.')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Alpha value for adversarial perturbation.')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta value for adversarial / clear switch (> beta / > clear).')
    parser.add_argument('--atack_iters', type=int, default=3, help='The number of iterations for the embedding adversarial attack.')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save the trained models.')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for the classification task.')

    args = parser.parse_args()
    main(args)