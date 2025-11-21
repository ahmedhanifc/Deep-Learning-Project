import matplotlib.pyplot as plt
import numpy as np

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