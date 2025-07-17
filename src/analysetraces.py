import json
import logging
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration du logger (comme dans votre code original)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def analyze_evaluation_traces_and_plot_confusion_matrices(traces_file_path: Path, output_dir: Path):
    """
    Analyzes the evaluation traces JSON file to count TP, FP, FN, TN
    for each entity code and plots confusion matrices as images.

    Args:
        traces_file_path (Path): The path to the JSON file containing evaluation traces.
        output_dir (Path): The directory where confusion matrix images will be saved.
    """
    if not traces_file_path.exists():
        print(f"Error: Traces file not found at {traces_file_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(traces_file_path, 'r', encoding='utf-8') as f:
            traces = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {traces_file_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {traces_file_path}: {e}")
        return

    # defaultdict will create a new dict for a key if it doesn't exist
    # and initialize counts to 0 for 'TP', 'FP', 'FN', 'TN'
    entity_code_counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})

    for trace in traces:
        status = trace.get('status')
        entity_code = trace.get('entity_code')

        if not status or not entity_code:
            # Skip entries that are malformed or not relevant for counting
            continue

        if status == 'TN':
            # TN entries have a 'count' field as per your trace generation code
            count = trace.get('count', 0)
            entity_code_counts[entity_code]['TN'] += count
        else:
            # For TP, FP, FN, each trace entry corresponds to one count
            entity_code_counts[entity_code][status] += 1

    # Generate and save confusion matrices
    print("\n--- Generating Confusion Matrices by Entity Code ---")
    if not entity_code_counts:
        print("No entity traces found for analysis.")
        return

    for code, counts in sorted(entity_code_counts.items()):
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        tn = counts['TN']


        conf_matrix = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d', # Format as integer
            cmap='Blues',
            cbar=False,
            xticklabels=['Prédit Négatif', 'Prédit Positif'],
            yticklabels=['Réel Négatif', 'Réel Positif'],
            linewidths=.5,
            linecolor='black'
        )
        plt.title(f'Matrice de Confusion pour {code}', fontsize=16)
        plt.xlabel('Label Prédit', fontsize=12)
        plt.ylabel('Label Réel', fontsize=12)
    
        # Save the plot
        image_filename = output_dir / f'confusion_matrix_{code}.png'
        plt.savefig(image_filename, bbox_inches='tight', dpi=300)
        plt.close() # Close the plot to free memory

        print(f"  Generated confusion matrix for '{code}' at {image_filename}")

    print("\n--- End of Report ---")

if __name__ == "__main__":

    traces_file = Path("./outputs/entity_traces/evaluation_tracesNew.json") 


    output_images_dir = Path("./confusion_matrices") 

    analyze_evaluation_traces_and_plot_confusion_matrices(traces_file, output_images_dir)