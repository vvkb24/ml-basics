"""
PyTorch Training Loop

Standard training loop pattern for PyTorch models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """Full training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
    
    return history


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    
    # Dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset[:80], batch_size=16)
    val_loader = DataLoader(dataset[80:], batch_size=16)
    
    print("Training loop ready to use!")
