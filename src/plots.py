import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_loss_curves(history, figsize=(10, 6)):
    """
    Plot training and testing loss curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_losses' key.
                 If 'test_losses' is present, will plot test loss as a curve.
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot training loss
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2, marker='o')
    
    # Plot test loss
    if 'test_losses' in history:
        plt.plot(epochs, history['test_losses'], 'r-', label='Test Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.ylim(bottom=0)  
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_curves(history, test_metrics=None, figsize=(10, 6)):
    """
    Plot training and testing accuracy curves vs epochs.
    
    Args:
        history: Dictionary containing training history with 'train_accuracies' key.
                 If 'test_accuracies' is present, will plot test accuracy as a curve.
        test_metrics: Optional dictionary containing test metrics with 'accuracy' key (deprecated, use history)
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_accuracies']) + 1)
    
    plt.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', 
             linewidth=2, marker='o')
    
    # Plot test accuracy from history if available
    if 'test_accuracies' in history:
        plt.plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy', linewidth=2, marker='s')
    # Fallback to test_metrics for backward compatibility
    elif test_metrics is not None and 'accuracy' in test_metrics:
        plt.axhline(y=test_metrics['accuracy'], color='r', linestyle='--', 
                   label=f'Test Accuracy: {test_metrics["accuracy"]:.4f}', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Test Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_combined_metrics(history, test_metrics=None, figsize=(12, 8)):
    """
    Plot multiple metrics (loss, accuracy, F1) in subplots.
    
    Args:
        history: Dictionary containing training history. If test metrics are present
                 (e.g., 'test_losses', 'test_accuracies'), they will be plotted as curves.
        test_metrics: Optional dictionary containing test metrics (deprecated, use history)
        figsize: Figure size tuple (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', 
                    linewidth=2, marker='o')
    if 'test_losses' in history:
        axes[0, 0].plot(epochs, history['test_losses'], 'r-', label='Test Loss', 
                       linewidth=2, marker='s')
    elif test_metrics is not None and 'loss' in test_metrics:
        axes[0, 0].axhline(y=test_metrics['loss'], color='r', linestyle='--', 
                          label=f'Test Loss: {test_metrics["loss"]:.4f}', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].set_ylim(bottom=0)  
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', 
                   linewidth=2, marker='o')
    if 'test_accuracies' in history:
        axes[0, 1].plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy', 
                       linewidth=2, marker='s')
    elif test_metrics is not None and 'accuracy' in test_metrics:
        axes[0, 1].axhline(y=test_metrics['accuracy'], color='r', linestyle='--', 
                          label=f'Test Accuracy: {test_metrics["accuracy"]:.4f}', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score 
    if 'train_f1_macro' in history:
        axes[1, 0].plot(epochs, history['train_f1_macro'], 'b-', label='Training F1', 
                       linewidth=2, marker='o')
        if 'test_f1_macro' in history:
            axes[1, 0].plot(epochs, history['test_f1_macro'], 'r-', label='Test F1', 
                           linewidth=2, marker='s')
        elif test_metrics is not None and 'f1_macro' in test_metrics:
            axes[1, 0].axhline(y=test_metrics['f1_macro'], color='r', linestyle='--', 
                              label=f'Test F1: {test_metrics["f1_macro"]:.4f}', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Precision 
    if 'train_precision_macro' in history:
        axes[1, 1].plot(epochs, history['train_precision_macro'], 'b-', 
                       label='Training Precision', linewidth=2, marker='o')
        if 'test_precision_macro' in history:
            axes[1, 1].plot(epochs, history['test_precision_macro'], 'r-', 
                           label='Test Precision', linewidth=2, marker='s')
        elif test_metrics is not None and 'precision_macro' in test_metrics:
            axes[1, 1].axhline(y=test_metrics['precision_macro'], color='r', linestyle='--', 
                              label=f'Test Precision: {test_metrics["precision_macro"]:.4f}', 
                              linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6), 
                         normalize=False, cmap='Blues'):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names
        figsize: Figure size tuple (width, height)
        normalize: If True, normalize the confusion matrix
        cmap: Colormap for the heatmap
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
    plt.show()
    
    return cm


def plot_training_history(history, test_metrics=None, figsize=(15, 10)):
    """
    Create a comprehensive plot of all training metrics.
    
    Args:
        history: Dictionary containing training history. If test metrics are present
                 (e.g., 'test_losses', 'test_accuracies'), they will be plotted as curves.
        test_metrics: Optional dictionary containing test metrics (deprecated, use history)
        figsize: Figure size tuple (width, height)
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
        elif test_metrics and 'loss' in test_metrics:
            axes[plot_idx].axhline(y=test_metrics['loss'], color='r', linestyle='--', 
                                  label=f'Test Loss', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Loss vs Epoch')
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
        elif test_metrics and 'accuracy' in test_metrics:
            axes[plot_idx].axhline(y=test_metrics['accuracy'], color='r', linestyle='--', 
                                  label=f'Test Accuracy', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Accuracy vs Epoch')
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
        elif test_metrics and 'f1_macro' in test_metrics:
            axes[plot_idx].axhline(y=test_metrics['f1_macro'], color='r', linestyle='--', 
                                 label=f'Test F1', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('F1 Score')
        axes[plot_idx].set_title('F1 Score vs Epoch')
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
        elif test_metrics and 'precision_macro' in test_metrics:
            axes[plot_idx].axhline(y=test_metrics['precision_macro'], color='r', linestyle='--', 
                                  label=f'Test Precision', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Precision')
        axes[plot_idx].set_title('Precision vs Epoch')
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
        elif test_metrics and 'recall_macro' in test_metrics:
            axes[plot_idx].axhline(y=test_metrics['recall_macro'], color='r', linestyle='--', 
                                  label=f'Test Recall', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Recall')
        axes[plot_idx].set_title('Recall vs Epoch')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

