import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report
import os

class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def main():
    # --- 1. Load Feature Data ---
    DATA_FILE = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/utils/nn_classifier_data/delay_features_data.npz'
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
        return

    print(f"Loading feature data from: {DATA_FILE}")
    data = np.load(DATA_FILE)
    X_data = data['X']
    y_data = data['y']
    
    # (Setup is the same...)
    INPUT_SIZE = X_data.shape[1]
    NUM_CLASSES = 3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    BATCH_SIZE = 64

    X_tensor = torch.from_numpy(X_data).float()
    y_tensor = torch.from_numpy(y_data).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MLPClassifier(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    print("\nStarting training...")
    
    # <-- NEW: Variable to track best performance
    best_accuracy = 0.0
    model_save_path = 'best_nn_classifier.pth'

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # --- Validation after each epoch ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')

        # <-- NEW: Save the model if it's the best one so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved to {model_save_path} with accuracy: {accuracy:.2f}%")

    print("\nTraining finished.")
    print(f"✅ Best model saved at '{model_save_path}' with final validation accuracy of {best_accuracy:.2f}%")

    # (Final evaluation can still be run to show the report)
    # ...

if __name__ == '__main__':
    main()