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
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

MODEL_ARCHITECTURE_CONFIG = {
    "input_size": 224 * 224 * 3,  # Flattened image size (height * width * channels)
    "hidden_sizes": [512, 256, 128],  # List of hidden layer sizes
    "num_classes": 4,  # Number of output classes
    "dropout_rates": [0.3, 0.2, 0.1],  # List of dropout rates for each hidden layer
}

