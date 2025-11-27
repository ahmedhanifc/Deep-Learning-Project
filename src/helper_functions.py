import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from datetime import datetime
import os
import torch
from src.config import MODEL_ARCHITECTURE_CONFIG
import json
import torch.optim as optim

def save_model(model, run_id, metrics, save_dir="models", save_best_only=True, best_metric_value=None):
    """
    Save model checkpoint.
    
    Args:
        model: The trained model
        run_id: Unique identifier for this run
        metrics: Dictionary with metrics (to include in checkpoint)
        save_dir: Directory to save models
        save_best_only: If True, only save if this is the best model
        best_metric_value: Current best metric value (for comparison)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'run_id': run_id,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'model_config': {
            'input_size': MODEL_ARCHITECTURE_CONFIG['input_size'],
            'hidden_sizes': MODEL_ARCHITECTURE_CONFIG['hidden_sizes'],
            'num_classes': MODEL_ARCHITECTURE_CONFIG['num_classes'],
            'dropout_rates': MODEL_ARCHITECTURE_CONFIG['dropout_rates'],
        }
    }
    
    if save_best_only and best_metric_value is not None:
        current_metric = metrics.get('test_accuracy', 0)
        if current_metric > best_metric_value:
            path = os.path.join(save_dir, f"best_model_{run_id}.pth")
            torch.save(checkpoint, path)
            print(f"Saved best model to {path}")
            return True, current_metric
    else:
        # Save all models
        path = os.path.join(save_dir, f"model_{run_id}.pth")
        torch.save(checkpoint, path)
        print(f"Saved model to {path}")
        return True, metrics.get('test_accuracy', 0)
    
    return False, best_metric_value


def plot_random_image_and_label(dataset, classes, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Plot a random image from the dataset with its label
    
    Args:
        dataset: The dataset to sample from
        classes: List of class names
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)
    '''
    
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]


    image = image.numpy()
    image = image.transpose((1, 2, 0))


    # denormalize
    mean = np.array(mean)
    std = np.array(std)


    image = image * std + mean
    image = np.clip(image, 0, 1)


    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.title(classes[label])
    plt.axis('off')
    plt.show()

    print(f"Index: {idx}")
    print(f"Label (integer): {label}")
    print(f"Class name: {dataset.classes[label]}")
    print(f"Image tensor shape: {image.shape}")


def print_model_parameter_breakdown(model, model_name="Model"):
    """
    Print a detailed step-by-step breakdown of parameter counts for each layer in a neural network.
    
    Args:
        model: PyTorch nn.Module instance
        model_name: Optional name for the model (for display purposes)
    """
    print("\n" + "=" * 90)
    print(f"DETAILED PARAMETER CALCULATION BREAKDOWN: {model_name}".center(90))
    print("=" * 90)
    
    layer_num = 1
    cumulative_total = 0
    
    # Iterate through each layer in the model
    for i, layer in enumerate(model.layers):
        # Get layer dimensions
        input_size = layer.in_features
        output_size = layer.out_features
        
        # Calculate parameters
        weights_count = input_size * output_size
        bias_count = output_size
        layer_params = weights_count + bias_count
        cumulative_total += layer_params
        
        # Get actual parameter tensors
        weight_param = layer.weight  # Shape: (output_size, input_size)
        bias_param = layer.bias       # Shape: (output_size,)
        
        # Print detailed breakdown
        print(f"\n{'─' * 90}")
        print(f"LAYER {layer_num}: Linear({input_size:,} → {output_size:,})")
        print(f"{'─' * 90}")
        print(f"  Input size:  {input_size:,}")
        print(f"  Output size: {output_size:,}")
        print()
        print(f"  WEIGHTS:")
        print(f"    • Shape: ({output_size:,}, {input_size:,})")
        print(f"    • Calculation: {input_size:,} × {output_size:,} = {weights_count:,}")
        print(f"    • Actual count: {weight_param.numel():,} ✓")
        print()
        print(f"  BIAS:")
        print(f"    • Shape: ({output_size:,},)")
        print(f"    • Calculation: {output_size:,} × 1 = {bias_count:,}")
        print(f"    • Actual count: {bias_param.numel():,} ✓")
        print()
        print(f"  LAYER TOTAL: {weights_count:,} + {bias_count:,} = {layer_params:,}")
        print(f"  Cumulative: {cumulative_total:,}")
        
        layer_num += 1
    
    print(f"\n{'=' * 90}")
    print(f"FINAL TOTAL: {cumulative_total:,} parameters")
    print(f"{'=' * 90}")
    
    # Verify
    actual = sum(p.numel() for p in model.parameters())
    print(f"Verification: Model has {actual:,} parameters")
    print()


def save_run_to_csv(run_data, csv_path="model_runs.csv"):
    """
    Save model run information to CSV file.
    
    Args:
        run_data: Dictionary containing run information
        csv_path: Path to CSV file
    """
    row = {
        'run_id': run_data.get('run_id', ''),
        'run_dir': run_data.get('run_dir', ''),  # Add run directory path
        'timestamp': run_data.get('timestamp', datetime.now().isoformat()),
        'total_time_seconds': run_data.get('total_time', 0),
        
        # Hyperparameters
        'learning_rate': run_data.get('hyperparameters', {}).get('learning_rate', ''),
        'batch_size': run_data.get('hyperparameters', {}).get('batch_size', ''),
        'epochs': run_data.get('hyperparameters', {}).get('epochs', ''),
        'hidden_sizes': str(run_data.get('hyperparameters', {}).get('hidden_sizes', '')),
        'dropout_rates': str(run_data.get('hyperparameters', {}).get('dropout_rates', '')),
        
        # Architecture
        'input_size': run_data.get('config', {}).get('input_size', ''),
        'num_classes': run_data.get('config', {}).get('num_classes', ''),
        
        # Training metrics (final epoch)
        'train_loss': run_data.get('train_metrics', {}).get('loss', ''),
        'train_accuracy': run_data.get('train_metrics', {}).get('accuracy', ''),
        'train_f1_macro': run_data.get('train_metrics', {}).get('f1_macro', ''),
        'train_precision_macro': run_data.get('train_metrics', {}).get('precision_macro', ''),
        'train_recall_macro': run_data.get('train_metrics', {}).get('recall_macro', ''),
        
        # Test metrics
        'test_loss': run_data.get('test_metrics', {}).get('loss', ''),
        'test_accuracy': run_data.get('test_metrics', {}).get('accuracy', ''),
        'test_f1_macro': run_data.get('test_metrics', {}).get('f1_macro', ''),
        'test_precision_macro': run_data.get('test_metrics', {}).get('precision_macro', ''),
        'test_recall_macro': run_data.get('test_metrics', {}).get('recall_macro', ''),
    }
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_model(model, run_dir, metrics, model_name="model.pth"):
    """
    Save model checkpoint to the run directory.
    
    Args:
        model: The trained model
        run_dir: Directory path for this run (e.g., "runs/run_1_20240101_120000")
        metrics: Dictionary with metrics (to include in checkpoint)
        model_name: Name for the model file (default: "model.pth")
    
    Returns:
        str: Path to saved model
    """
    os.makedirs(run_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'model_config': {
            'input_size': MODEL_ARCHITECTURE_CONFIG['input_size'],
            'hidden_sizes': MODEL_ARCHITECTURE_CONFIG['hidden_sizes'],
            'num_classes': MODEL_ARCHITECTURE_CONFIG['num_classes'],
            'dropout_rates': MODEL_ARCHITECTURE_CONFIG['dropout_rates'],
        }
    }
    
    model_path = os.path.join(run_dir, model_name)
    torch.save(checkpoint, model_path)
    print(f"Saved model to {model_path}")
    return model_path


def save_config_to_json(config_dict, run_dir, filename="config.json"):
    """
    Save configuration to JSON file in run directory.
    
    Args:
        config_dict: Dictionary containing configuration
        run_dir: Directory path for this run
        filename: Name for the config file
    """
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, filename)
    
    # Convert any non-serializable types to strings
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Saved config to {config_path}")

def get_optimizer(model_parameters, config):
    opt_type = config["optimizer"]["type"]
    lr = config["optimizer"]["learning_rate"]
    wd = config["optimizer"]["weight_decay"] 
    
    print(f"[Setup] Using {opt_type} optimizer with LR={lr} and L2={wd}")

    if opt_type == "Adam":
        return optim.Adam(model_parameters, lr=lr, weight_decay=wd)
    elif opt_type == "SGD":
        momentum = config["optimizer"].get("momentum", 0.9)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_type == "RMSprop":
        return optim.RMSprop(model_parameters, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")