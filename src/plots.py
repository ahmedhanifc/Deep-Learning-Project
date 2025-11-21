def plot_training_curves(history, figsize=(15, 5)):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary returned from train_model() containing:
            - 'train_losses', 'test_losses', 'train_accuracies', 'test_accuracies'
        figsize: Figure size tuple (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_losses'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy vs Epoch', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, figsize=(10, 8), normalize=False):
    """
    Plot a confusion matrix with optional normalization.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        figsize: Figure size tuple
        normalize: If True, normalize the confusion matrix (show percentages)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_class_metrics(metrics, class_names, figsize=(15, 5)):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        metrics: Dictionary returned from compute_classification_metrics()
        class_names: List of class names
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # Precision
    axes[0].bar(x, metrics['precision_per_class'], width, label='Precision', color='skyblue', edgecolor='black')
    axes[0].axhline(y=metrics['precision_macro'], color='r', linestyle='--', 
                    label=f'Macro Avg: {metrics["precision_macro"]:.3f}')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])
    
    # Recall
    axes[1].bar(x, metrics['recall_per_class'], width, label='Recall', color='lightgreen', edgecolor='black')
    axes[1].axhline(y=metrics['recall_macro'], color='r', linestyle='--', 
                    label=f'Macro Avg: {metrics["recall_macro"]:.3f}')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall per Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.1])
    
    # F1-Score
    axes[2].bar(x, metrics['f1_per_class'], width, label='F1-Score', color='salmon', edgecolor='black')
    axes[2].axhline(y=metrics['f1_macro'], color='r', linestyle='--', 
                    label=f'Macro Avg: {metrics["f1_macro"]:.3f}')
    axes[2].set_xlabel('Class', fontsize=12)
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()


def print_metrics_summary(metrics, class_names):
    """
    Print a comprehensive summary of all classification metrics.
    
    Args:
        metrics: Dictionary returned from compute_classification_metrics()
        class_names: List of class names
    """
    print("=" * 80)
    print("CLASSIFICATION METRICS SUMMARY".center(80))
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}")
    
    print("\n" + "-" * 80)
    print("AVERAGE METRICS:")
    print("-" * 80)
    print(f"Macro Average:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT:")
    print("=" * 80)
    print(metrics['classification_report'])