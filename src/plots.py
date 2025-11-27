import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_loss_curves(history, figsize=(10, 6), save_path=None, show=True, ylim=None):
    """
    Plot training and testing loss curves vs epochs.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size tuple
        save_path: If provided, save plot to this path (e.g., 'plots/run_1_loss.png')
        show: Whether to display the plot
        ylim: Tuple (min, max) for y-axis limits. If None, uses (0, auto)
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2, marker='o')
    
    if 'test_losses' in history:
        plt.plot(epochs, history['test_losses'], 'r-', label='Test Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss vs Epoch', fontsize=14, fontweight='bold')
    
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(bottom=0)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_curves(history, figsize=(10, 6), save_path=None, show=True, ylim=(0, 1)):
    """
    Plot training and testing accuracy curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_accuracies' key.
                 If 'test_accuracies' is present, will plot test accuracy as a curve.
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        ylim: Tuple (min, max) for y-axis limits. Default is (0, 1)
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_accuracies']) + 1)
    
    plt.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', 
             linewidth=2, marker='o')
    
    if 'test_accuracies' in history:
        plt.plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Test Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.ylim(ylim)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_f1_curves(history, figsize=(10, 6), save_path=None, show=True, ylim=(0, 1)):
    """
    Plot training and testing F1 score curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_f1_macro' key.
                 If 'test_f1_macro' is present, will plot test F1 as a curve.
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        ylim: Tuple (min, max) for y-axis limits. Default is (0, 1)
    """
    if 'train_f1_macro' not in history:
        print("Warning: 'train_f1_macro' not found in history. Skipping F1 plot.")
        return
    
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_f1_macro']) + 1)
    
    plt.plot(epochs, history['train_f1_macro'], 'b-', label='Training F1', 
             linewidth=2, marker='o')
    
    if 'test_f1_macro' in history:
        plt.plot(epochs, history['test_f1_macro'], 'r-', label='Test F1', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Test F1 Score vs Epoch', fontsize=14, fontweight='bold')
    plt.ylim(ylim)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_curves(history, figsize=(10, 6), save_path=None, show=True, ylim=(0, 1)):
    """
    Plot training and testing precision curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_precision_macro' key.
                 If 'test_precision_macro' is present, will plot test precision as a curve.
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        ylim: Tuple (min, max) for y-axis limits. Default is (0, 1)
    """
    if 'train_precision_macro' not in history:
        print("Warning: 'train_precision_macro' not found in history. Skipping precision plot.")
        return
    
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_precision_macro']) + 1)
    
    plt.plot(epochs, history['train_precision_macro'], 'b-', label='Training Precision', 
             linewidth=2, marker='o')
    
    if 'test_precision_macro' in history:
        plt.plot(epochs, history['test_precision_macro'], 'r-', label='Test Precision', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Training and Test Precision vs Epoch', fontsize=14, fontweight='bold')
    plt.ylim(ylim)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_recall_curves(history, figsize=(10, 6), save_path=None, show=True, ylim=(0, 1)):
    """
    Plot training and testing recall curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_recall_macro' key.
                 If 'test_recall_macro' is present, will plot test recall as a curve.
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        ylim: Tuple (min, max) for y-axis limits. Default is (0, 1)
    """
    if 'train_recall_macro' not in history:
        print("Warning: 'train_recall_macro' not found in history. Skipping recall plot.")
        return
    
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_recall_macro']) + 1)
    
    plt.plot(epochs, history['train_recall_macro'], 'b-', label='Training Recall', 
             linewidth=2, marker='o')
    
    if 'test_recall_macro' in history:
        plt.plot(epochs, history['test_recall_macro'], 'r-', label='Test Recall', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Training and Test Recall vs Epoch', fontsize=14, fontweight='bold')
    plt.ylim(ylim)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_metrics(history, figsize=(12, 8), save_path=None, show=True, 
                         loss_ylim=None, metric_ylim=(0, 1)):
    """
    Plot multiple metrics (loss, accuracy, F1, precision) in subplots.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        loss_ylim: Tuple (min, max) for loss y-axis. If None, uses (0, auto)
        metric_ylim: Tuple (min, max) for metric y-axes. Default is (0, 1)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', 
                    linewidth=2, marker='o')
    if 'test_losses' in history:
        axes[0, 0].plot(epochs, history['test_losses'], 'r-', label='Test Loss', 
                       linewidth=2, marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Epoch')
    if loss_ylim:
        axes[0, 0].set_ylim(loss_ylim)
    else:
        axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', 
                   linewidth=2, marker='o')
    if 'test_accuracies' in history:
        axes[0, 1].plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy', 
                       linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].set_ylim(metric_ylim)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score 
    if 'train_f1_macro' in history:
        axes[1, 0].plot(epochs, history['train_f1_macro'], 'b-', label='Training F1', 
                       linewidth=2, marker='o')
        if 'test_f1_macro' in history:
            axes[1, 0].plot(epochs, history['test_f1_macro'], 'r-', label='Test F1', 
                           linewidth=2, marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Epoch')
        axes[1, 0].set_ylim(metric_ylim)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Precision 
    if 'train_precision_macro' in history:
        axes[1, 1].plot(epochs, history['train_precision_macro'], 'b-', 
                       label='Training Precision', linewidth=2, marker='o')
        if 'test_precision_macro' in history:
            axes[1, 1].plot(epochs, history['test_precision_macro'], 'r-', 
                           label='Test Precision', linewidth=2, marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Epoch')
        axes[1, 1].set_ylim(metric_ylim)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6), 
                         normalize=False, cmap='Blues', save_path=None, show=True):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names
        figsize: Figure size tuple (width, height)
        normalize: If True, normalize the confusion matrix
        cmap: Colormap for the heatmap
        save_path: If provided, save plot to this path
        show: Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    if class_names is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [f'Class {i}' for i in unique_classes]
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, 
                yticklabels=class_names, cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm


def plot_training_history(history, figsize=(15, 10), save_path=None, show=True,
                         loss_ylim=None, metric_ylim=(0, 1)):
    """
    Create a comprehensive plot of all training metrics.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size tuple (width, height)
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        loss_ylim: Tuple (min, max) for loss y-axis. If None, uses (0, auto)
        metric_ylim: Tuple (min, max) for metric y-axes. Default is (0, 1)
    """
    num_metrics = sum([
        'train_losses' in history,
        'train_accuracies' in history,
        'train_f1_macro' in history,
        'train_precision_macro' in history,
        'train_recall_macro' in history
    ])
    
    if num_metrics == 0:
        print("No training history data found!")
        return
    
    rows = (num_metrics + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    epochs = range(1, len(history['train_losses']) + 1)
    plot_idx = 0
    
    # Loss
    if 'train_losses' in history:
        axes[plot_idx].plot(epochs, history['train_losses'], 'b-', 
                           label='Training Loss', linewidth=2, marker='o')
        if 'test_losses' in history:
            axes[plot_idx].plot(epochs, history['test_losses'], 'r-', 
                               label='Test Loss', linewidth=2, marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Loss vs Epoch')
        if loss_ylim:
            axes[plot_idx].set_ylim(loss_ylim)
        else:
            axes[plot_idx].set_ylim(bottom=0)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Accuracy
    if 'train_accuracies' in history:
        axes[plot_idx].plot(epochs, history['train_accuracies'], 'b-', 
                           label='Training Accuracy', linewidth=2, marker='o')
        if 'test_accuracies' in history:
            axes[plot_idx].plot(epochs, history['test_accuracies'], 'r-', 
                               label='Test Accuracy', linewidth=2, marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Accuracy vs Epoch')
        axes[plot_idx].set_ylim(metric_ylim)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # F1 Score
    if 'train_f1_macro' in history:
        axes[plot_idx].plot(epochs, history['train_f1_macro'], 'b-', 
                           label='Training F1', linewidth=2, marker='o')
        if 'test_f1_macro' in history:
            axes[plot_idx].plot(epochs, history['test_f1_macro'], 'r-', 
                               label='Test F1', linewidth=2, marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('F1 Score')
        axes[plot_idx].set_title('F1 Score vs Epoch')
        axes[plot_idx].set_ylim(metric_ylim)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Precision
    if 'train_precision_macro' in history:
        axes[plot_idx].plot(epochs, history['train_precision_macro'], 'b-', 
                          label='Training Precision', linewidth=2, marker='o')
        if 'test_precision_macro' in history:
            axes[plot_idx].plot(epochs, history['test_precision_macro'], 'r-', 
                               label='Test Precision', linewidth=2, marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Precision')
        axes[plot_idx].set_title('Precision vs Epoch')
        axes[plot_idx].set_ylim(metric_ylim)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Recall
    if 'train_recall_macro' in history:
        axes[plot_idx].plot(epochs, history['train_recall_macro'], 'b-', 
                          label='Training Recall', linewidth=2, marker='o')
        if 'test_recall_macro' in history:
            axes[plot_idx].plot(epochs, history['test_recall_macro'], 'r-', 
                               label='Test Recall', linewidth=2, marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Recall')
        axes[plot_idx].set_title('Recall vs Epoch')
        axes[plot_idx].set_ylim(metric_ylim)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

