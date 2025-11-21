import matplotlib.pyplot as plt
import numpy as np

def plot_random_image_and_label(dataset, classes):
    '''
    Plot a random image from the dataset with its label
    '''
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]


    image = image.numpy()
    image = image.transpose((1, 2, 0))


    # denormalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


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
