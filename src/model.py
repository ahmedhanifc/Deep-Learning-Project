import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes: list[int], num_classes: int, dropout_rates: list[float] = None):
        super(NeuralNetwork, self).__init__()

        # Build complete list of layer sizes from input to output
        # [input_size] creates a list with one element: [150528]
        # + hidden_sizes adds hidden layers: [150528, 512, 256, 128]
        # + [num_classes] adds output layer: [150528, 512, 256, 128, 4]
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        #layer_sizes[0]=150528, layer_sizes[1]=512 creates Linear(150528, 512)
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.dropouts = nn.ModuleList()

        # Why different rates? Early layers learn low-level features (need more regularization)
        # Later layers learn high-level features (need less regularization)
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_sizes)

        for rate in dropout_rates:
            if rate > 0:
                self.dropouts.append(nn.Dropout(rate))
            else:
                self.dropouts.append(nn.Identity())

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten the image

        # process all layers except last because that will be processed separately because its the output layer
        for i in range(len(self.layers)-1):

            # linear transformation: x = W.T * x + b
            x = self.layers[i](x) # # This is equivalent to: x = x @ W.T + b (wow)
            x = F.relu(x)
            x = self.dropouts[i](x)

        # Last Layer
        x = self.layers[-1](x)
        return x    


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch with multiclass metrics.
    
    Returns:
        dict: {'loss': avg_loss, 'accuracy': accuracy, 
               'precision_macro': precision, 'recall_macro': recall, 
               'f1_macro': f1, 'precision_weighted': precision_w, 
               'recall_weighted': recall_w, 'f1_weighted': f1_w,
               'y_true': true_labels, 'y_pred': predicted_labels}
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Collect predictions and labels
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'y_true': y_true,
        'y_pred': y_pred
    }

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset with multiclass metrics.
    
    Returns:
        dict: Same structure as train_epoch
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # conversion to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'loss': running_loss / len(data_loader),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'y_true': y_true,
        'y_pred': y_pred
    }


def train_model(model, train_loader, criterion, optimizer, device, num_epochs, test_loader=None, verbose=True, scheduler=None):
    """
    Train model with separate train and validation phases.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of epochs to train
        test_loader: Optional DataLoader for test data (if provided, will evaluate each epoch)
        verbose: Whether to print progress
    
    Returns:
        dict: Training history with metrics per epoch
    """
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'train_precision_macro': [],
        'train_recall_macro': [],
        'train_f1_macro': [],
        'train_precision_weighted': [],
        'train_recall_weighted': [],
        'train_f1_weighted': [],
    }
    
    if test_loader is not None:
        history['test_losses'] = []
        history['test_accuracies'] = []
        history['test_precision_macro'] = []
        history['test_recall_macro'] = []
        history['test_f1_macro'] = []
        history['test_precision_weighted'] = []
        history['test_recall_weighted'] = []
        history['test_f1_weighted'] = []
    
    for epoch in range(num_epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_losses'].append(train_metrics['loss'])
        history['train_accuracies'].append(train_metrics['accuracy'])
        history['train_precision_macro'].append(train_metrics['precision_macro'])
        history['train_recall_macro'].append(train_metrics['recall_macro'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['train_precision_weighted'].append(train_metrics['precision_weighted'])
        history['train_recall_weighted'].append(train_metrics['recall_weighted'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])

        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Test
        if test_loader is not None:
            test_metrics = evaluate_model(model, test_loader, criterion, device)
            history['test_losses'].append(test_metrics['loss'])
            history['test_accuracies'].append(test_metrics['accuracy'])
            history['test_precision_macro'].append(test_metrics['precision_macro'])
            history['test_recall_macro'].append(test_metrics['recall_macro'])
            history['test_f1_macro'].append(test_metrics['f1_macro'])
            history['test_precision_weighted'].append(test_metrics['precision_weighted'])
            history['test_recall_weighted'].append(test_metrics['recall_weighted'])
            history['test_f1_weighted'].append(test_metrics['f1_weighted'])

            if scheduler is not None:
                scheduler.step(test_metrics['loss'])
        
        # Print progress
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}, F1: {train_metrics["f1_macro"]:.4f}, Precision: {train_metrics["precision_macro"]:.4f}, Recall: {train_metrics["recall_macro"]:.4f}')
            if test_loader is not None:
                print(f'  Test  - Loss: {test_metrics["loss"]:.4f}, Acc: {test_metrics["accuracy"]:.4f}, F1: {test_metrics["f1_macro"]:.4f}, Precision: {test_metrics["precision_macro"]:.4f}, Recall: {test_metrics["recall_macro"]:.4f}')
            print('-' * 50)
    
    return history