import torch

TRANSFORM_CONFIG = {
    "image_size": (224, 224),  # (height, width)
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "normalize_std": [0.229, 0.224, 0.225],   # ImageNet std
    "random_horizontal_flip_prob": 0.5,
    "random_rotation_degrees": 15,
}
MODEL_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    'weight_decay': 0.01,  # L2 regularization

    #----------------------------
    "patience": 5,  # epochs to wait after no improvement
    "delta": 0.01,  # minimum change in the monitored metric
    "best_val_loss": float("inf"),  # best validation loss to compare against
    "no_improvement_count": 0  # count of epochs with no improvement
}

MODEL_ARCHITECTURE_CONFIG = {
    "input_size": 224 * 224 * 3,  # Flattened image size (height * width * channels)
    "hidden_sizes": [512, 256, 128], 
    "num_classes": 4,  # Number of output classes
    "dropout_rates": [0.3, 0.2, 0.1],  
}



