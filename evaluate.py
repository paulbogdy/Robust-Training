import os
import pandas as pd
import numpy as np
import argparse

def calculate_metrics(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Calculate Clean Accuracy (accuracy on original sentences)
    clean_accuracy = (df['Pred_original'] == df['True']).mean() * 100
    
    # Calculate Perturbed Accuracy (accuracy on perturbed sentences)
    perturbed_accuracy = (df['Pred_perturbed'] == df['True']).mean() * 100
    
    # Filter out samples where Pred_original != True
    filtered_df = df[df['Pred_original'] == df['True']]
    
    # Calculate Attack Success Rate (ASR) on filtered samples
    asr = (filtered_df['success'] == True).mean() * 100

    # Calculate average character and token edit distances, and their standard deviations on filtered samples
    avg_char_edit_dist = filtered_df['Dist_char'].mean()
    std_char_edit_dist = filtered_df['Dist_char'].std()
    avg_token_edit_dist = filtered_df['Dist_token'].mean()
    std_token_edit_dist = filtered_df['Dist_token'].std()
    
    # Calculate average similarity and standard deviation on filtered samples
    avg_similarity = filtered_df['similarity'].mean()
    std_similarity = filtered_df['similarity'].std()
    
    # Calculate average run time and standard deviation on filtered samples
    avg_runtime = filtered_df['time'].mean()
    std_runtime = filtered_df['time'].std()
    
    # Format the values with ± for standard deviations
    char_edit_dist = f"{avg_char_edit_dist:.2f} ± {std_char_edit_dist:.2f}"
    token_edit_dist = f"{avg_token_edit_dist:.2f} ± {std_token_edit_dist:.2f}"
    similarity = f"{avg_similarity:.2f} ± {std_similarity:.2f}"
    runtime = f"{avg_runtime:.2f} ± {std_runtime:.2f}"
    
    return {
        "Clean Acc (%)": f"{clean_accuracy:.2f}%",
        "Pert Acc (%)": f"{perturbed_accuracy:.2f}%",
        "ASR (%)": f"{asr:.2f}%",
        "Avg. Char Edit Dist": char_edit_dist,
        "Avg. Token Edit Dist": token_edit_dist,
        "Avg. Similarity": similarity,
        "Avg. Run Time": runtime
    }

def summarize_attack_folder(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            attack_name = file_name.replace(".csv", "")
            file_path = os.path.join(folder_path, file_name)
            
            metrics = calculate_metrics(file_path)
            metrics['Attack'] = attack_name
            
            results.append(metrics)

    # Convert results to DataFrame
    summary_df = pd.DataFrame(results)
    summary_df = summary_df[['Attack', 'Clean Acc (%)', 'Pert Acc (%)', 'ASR (%)', 'Avg. Char Edit Dist', 'Avg. Token Edit Dist', 'Avg. Similarity', 'Avg. Run Time']]
    
    return summary_df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Summarize attack metrics from CSV files in a folder.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing CSV files.")
    
    # Parse arguments
    args = parser.parse_args()
    folder_path = args.folder_path

    # Generate and display the summary table
    summary_table = summarize_attack_folder(folder_path)
    print(summary_table)
