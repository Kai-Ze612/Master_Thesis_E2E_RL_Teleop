import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _, (h_n, _) = self.lstm(x, (h0, c0))
        out = self.fc(h_n[-1, :, :])
        return out

def main():
    # --- 1. Load and Prepare Data ---
    DATA_FILE = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/utils/lstm_classifier_data/delay_training_data.npz'

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
        return

    print(f"Loading data from: {DATA_FILE}")
    data = np.load(DATA_FILE)
    X_data = data['X']
    y_data = data['y']

    # --- Setup ---
    X_tensor = torch.from_numpy(X_data).float()
    y_tensor = torch.from_numpy(y_data).long()
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    BATCH_SIZE = 64
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\nStarting training...")
    
    # <-- NEW: Variable to track best performance
    best_accuracy = 0.0
    model_save_path = 'best_lstm_classifier.pth' # Use a different name for the LSTM model

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
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
    print(f"✅ Best LSTM model saved at '{model_save_path}' with final validation accuracy of {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()