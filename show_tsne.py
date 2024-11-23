import os
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse

# Load the saved model
def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# Extract embeddings for a dataset
def extract_embeddings(model, tokenizer, dataset, max_length=128, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    embeddings = []
    labels = []

    # Process the dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        texts = batch['sentence']
        label = batch['label']
        
        # Tokenize inputs
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Extract the last hidden state
            cls_embeddings = hidden_states[:, 0, :]  # Use the [CLS] token embedding

        embeddings.append(cls_embeddings.cpu())  # Move to CPU for further processing
        labels.extend(label)

    # Concatenate all batches
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, labels

# Perform T-SNE and plot
def plot_tsne(embeddings, labels, save_path, num_classes=2):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in range(num_classes):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            reduced_embeddings[indices, 0],
            reduced_embeddings[indices, 1],
            label=f"Class {label}",
            alpha=0.6
        )

    plt.title("T-SNE Visualization of Embeddings")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.legend()

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plot_file = os.path.join(save_path, "tsne_plot.png")
    plt.savefig(plot_file)
    print(f"T-SNE plot saved to {plot_file}")
    plt.close()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-SNE visualization for embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--output_folder", type=str, default="results", help="Folder to save the T-SNE plot")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples from the dataset to use")
    args = parser.parse_args()

    # Load model and tokenizer
    model_path = args.model_path
    output_folder = os.path.join(model_path, args.output_folder)
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load SST dataset
    print("Loading SST2 dataset...")
    dataset = load_dataset("glue", "sst2")["train"]
    dataset = dataset.select(range(args.dataset_size))  # Optionally, reduce size for T-SNE visualization

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, tokenizer, dataset)

    # Perform T-SNE and plot
    print("Performing T-SNE...")
    plot_tsne(embeddings.numpy(), labels, save_path=output_folder)
