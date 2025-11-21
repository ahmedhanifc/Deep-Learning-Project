import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Calculate confusion matrix for classification results.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names for display
    
    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print a detailed classification report.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=False
    )
    print(report)
    return report


def calculate_metrics_summary(y_true, y_pred):
    """
    Calculate a comprehensive summary of metrics.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def print_metrics_summary(y_true, y_pred, dataset_name="Dataset"):
    """
    Print a formatted summary of all metrics.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        dataset_name: Name of the dataset (for display purposes)
    """
    metrics = calculate_metrics_summary(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Metrics Summary for {dataset_name}")
    print(f"{'='*60}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def calculate_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names
    
    Returns:
        dict: Dictionary with per-class metrics
    """
    if class_names is None:
        unique_classes = np.unique(y_true)
        class_names = [f'Class {i}' for i in unique_classes]
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        # Create binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return per_class_metrics


def print_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Print per-class metrics in a formatted table.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Optional list of class names
    """
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred, class_names)
    
    print(f"\n{'='*60}")
    print("Per-Class Metrics")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print(f"{'-'*60}")
    
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print(f"{'='*60}\n")
    
    return per_class_metrics

