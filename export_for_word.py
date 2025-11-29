"""
Script to combine training curve images with hyperparameters caption for Word document export.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np


def load_config(config_path):
    """Load hyperparameters from config.json file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_f1_scores(csv_path, run_id):
    """
    Load train and test F1 scores from CSV file.
    
    Args:
        csv_path: Path to model_runs.csv
        run_id: Run ID to look up (e.g., 'run_2_20251122_020743')
        
    Returns:
        Tuple of (train_f1, test_f1) or (None, None) if not found
    """
    try:
        df = pd.read_csv(csv_path)
        row = df[df['run_id'] == run_id]
        if not row.empty:
            train_f1 = row['train_f1_macro'].iloc[0]
            test_f1 = row['test_f1_macro'].iloc[0]
            return train_f1, test_f1
    except Exception as e:
        print(f"Warning: Could not load F1 scores from CSV: {e}")
    return None, None


def format_hyperparameters(config, train_f1=None, test_f1=None):
    """
    Format hyperparameters into a readable caption string.
    
    Args:
        config: Dictionary containing config data
        train_f1: Training F1 score (optional)
        test_f1: Test F1 score (optional)
        
    Returns:
        Formatted caption string
    """
    hp = config.get('hyperparameters', {})
    
    # Extract key hyperparameters
    name = hp.get('name', 'N/A')
    lr = hp.get('learning_rate', 'N/A')
    batch_size = hp.get('batch_size', 'N/A')
    epochs = hp.get('epochs', 'N/A')
    hidden_sizes = hp.get('hidden_sizes', [])
    dropout_rates = hp.get('dropout_rates', [])
    
    # Format hidden sizes
    if hidden_sizes:
        hidden_str = ' → '.join(map(str, hidden_sizes))
    else:
        hidden_str = 'N/A'
    
    # Format dropout rates
    if dropout_rates:
        if len(set(dropout_rates)) == 1:
            dropout_str = f"{dropout_rates[0]} (all layers)"
        else:
            dropout_str = ', '.join(map(str, dropout_rates))
    else:
        dropout_str = 'N/A'
    
    # Create formatted caption
    caption_lines = [
        f"Model Configuration: {name}",
        "",
        f"Learning Rate: {lr}",
        f"Batch Size: {batch_size}",
        f"Epochs: {epochs}",
        f"Hidden Layer Sizes: {hidden_str}",
        f"Dropout Rates: {dropout_str}"
    ]
    
    # Add F1 scores if available
    if train_f1 is not None and test_f1 is not None:
        caption_lines.append("")
        caption_lines.append(f"Train F1 Score: {train_f1:.4f}")
        caption_lines.append(f"Test F1 Score: {test_f1:.4f}")
    
    return '\n'.join(caption_lines)


def combine_images_with_caption(loss_image_path, f1_image_path, config_path, output_path, 
                                csv_path=None, run_id=None):
    """
    Combine two training curve images side-by-side with hyperparameters caption.
    
    Args:
        loss_image_path: Path to loss curve image
        f1_image_path: Path to F1 curve image
        config_path: Path to config.json file
        output_path: Path to save the combined image
        csv_path: Optional path to model_runs.csv to get F1 scores
        run_id: Optional run_id to look up F1 scores in CSV
    """
    # Load images
    loss_img = mpimg.imread(loss_image_path)
    f1_img = mpimg.imread(f1_image_path)
    
    # Load config
    config = load_config(config_path)
    
    # Load F1 scores if CSV path and run_id are provided
    train_f1, test_f1 = None, None
    if csv_path and run_id:
        train_f1, test_f1 = load_f1_scores(csv_path, run_id)
    
    # Format caption with hyperparameters and F1 scores
    caption_text = format_hyperparameters(config, train_f1, test_f1)
    
    # Get image dimensions
    loss_height, loss_width = loss_img.shape[:2]
    f1_height, f1_width = f1_img.shape[:2]
    
    # Use a standard figure size suitable for Word documents (wide format)
    # Calculate aspect ratio to maintain image quality
    max_height = max(loss_height, f1_height)
    total_width = loss_width + f1_width
    aspect_ratio = total_width / max_height
    
    # Create figure with appropriate size (width in inches, maintaining aspect)
    fig_width = 16  # Good width for Word documents
    fig_height = (fig_width / aspect_ratio) + 2  # Add space for caption
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    gs = gridspec.GridSpec(2, 2, height_ratios=[fig_height-2, 2], width_ratios=[1, 1], 
                          hspace=0.4, wspace=0.15)
    
    # Display loss curve (left)
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(loss_img)
    ax1.axis('off')
    ax1.set_title('Training and Test Loss vs Epoch', fontsize=13, pad=15, fontweight='bold')
    
    # Display F1 curve (right)
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(f1_img)
    ax2.axis('off')
    ax2.set_title('Training and Test F1 Score vs Epoch', fontsize=13, pad=15, fontweight='bold')
    
    # Add caption spanning both columns
    ax_caption = plt.subplot(gs[1, :])
    ax_caption.axis('off')
    ax_caption.text(0.5, 0.5, caption_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=11,
                   family='sans-serif',
                   transform=ax_caption.transAxes,
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.5, edgecolor='black', linewidth=1.5))
    
    # Save with high DPI for Word document quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Combined image saved to: {output_path}")


def main():
    """Main function to run the export."""
    # Define paths
    run_dir = "runs/run_1_20251129_201029"
    run_id = "run_1_20251129_201029"
    loss_image_path = os.path.join(run_dir, "loss_curve.png")
    f1_image_path = os.path.join(run_dir, "f1_curve.png")
    config_path = os.path.join(run_dir, "config.json")
    csv_path = "model_runs.csv"
    output_path = os.path.join(run_dir, "combined_curves_with_hyperparams.png")
    
    # Check if files exist
    if not os.path.exists(loss_image_path):
        raise FileNotFoundError(f"Loss curve image not found: {loss_image_path}")
    if not os.path.exists(f1_image_path):
        raise FileNotFoundError(f"F1 curve image not found: {f1_image_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Combine images with caption (including F1 scores from CSV if available)
    combine_images_with_caption(loss_image_path, f1_image_path, config_path, output_path,
                                csv_path=csv_path, run_id=run_id)


if __name__ == "__main__":
    main()

