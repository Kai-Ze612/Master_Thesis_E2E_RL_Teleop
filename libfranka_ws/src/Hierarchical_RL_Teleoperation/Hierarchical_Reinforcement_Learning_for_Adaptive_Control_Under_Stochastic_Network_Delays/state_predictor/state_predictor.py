#!/usr/bin/env python3
"""
Single LSTM Model Training Script
Train one model at a time for better resource management and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import time
from datetime import datetime

class PositionLSTM(nn.Module):
    def __init__(self, 
                 input_size=3,      # [x, y, z] positions
                 hidden_size=128,   # LSTM hidden dimension
                 num_layers=2,      # Number of LSTM layers
                 dropout=0.2):      # Dropout for regularization
        super().__init__()
       
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, input_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMTrainer:
    """Training class for LSTM position predictor"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu', learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, epoch_num, total_epochs):
        """Train for one epoch with progress tracking"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar for the epoch
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {epoch_num+1}/{total_epochs} [Training]',
                   leave=True,
                   dynamic_ncols=True)
        
        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar with current loss
            current_avg = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{current_avg:.6f}'})
        
        pbar.close()
        epoch_loss = total_loss / num_batches
        return epoch_loss
    
    def validate(self, epoch_num, total_epochs):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader,
                   desc=f'Epoch {epoch_num+1}/{total_epochs} [Validation]',
                   leave=True,
                   dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_inputs, batch_targets in pbar:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                current_avg = total_loss / num_batches
                pbar.set_postfix({'Val Loss': f'{current_avg:.6f}'})
        
        pbar.close()
        val_loss = total_loss / num_batches
        return val_loss
    
    def train(self, num_epochs, save_path, model_name, patience=15, min_delta=1e-5):
        """Full training loop"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Early stopping variables
        early_stopping_counter = 0
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Dataset size: {len(self.train_loader.dataset):,} training samples")
        print(f"  Batches per epoch: {len(self.train_loader):,}")
        print(f"  Max epochs: {num_epochs}")
        print(f"  Patience: {patience}")
        print(f"  Min delta: {min_delta}")
        print(f"  Device: {self.device}")
        print(f"  Save path: {save_path}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch, num_epochs)
            self.val_losses.append(val_loss)
            
            # Calculate times
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"Results: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.6f}")
            print(f"Time: Epoch={epoch_time:.1f}s, Total={elapsed_time/60:.1f}min")
            
            # Early stopping check
            if val_loss < self.best_val_loss - min_delta:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': self.best_val_loss,
                    'model_config': {
                        'input_size': self.model.input_size,
                        'hidden_size': self.model.hidden_size,
                        'num_layers': self.model.num_layers
                    }
                }, os.path.join(save_path, f'{model_name}_best.pth'))
                
                print(f"★ NEW BEST MODEL SAVED! (improvement: {improvement:.6f})")
                
            else:
                early_stopping_counter += 1
                print(f"No improvement for {early_stopping_counter}/{patience} epochs")
                if early_stopping_counter >= patience:
                    print(f"EARLY STOPPING triggered after {patience} epochs without improvement")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"TRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs trained: {len(self.train_losses)}")
        print(f"Model saved to: {os.path.join(save_path, f'{model_name}_best.pth')}")
        print("="*60)
        
        return self.best_val_loss
    
    def plot_training_history(self, save_path, model_name):
        """Plot and save training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.axhline(y=self.best_val_loss, color='green', linestyle='--', 
                   label=f'Best Val Loss: {self.best_val_loss:.6f}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_name}_training_history.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()

def load_and_prepare_data(data_path, batch_size=512):
    """Load and prepare data for training"""
    print(f"Loading data from: {data_path}")
    
    # Check if files exist
    train_file = os.path.join(data_path, 'train_data.npy')
    val_file = os.path.join(data_path, 'val_data.npy')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation data not found: {val_file}")
    
    # Load train and validation data
    train_data = np.load(train_file, allow_pickle=True).item()
    val_data = np.load(val_file, allow_pickle=True).item()
    
    # Extract sequences and targets
    train_sequences = train_data['input_sequences']
    train_targets = train_data['target_positions']
    
    val_sequences = val_data['input_sequences']
    val_targets = val_data['target_positions']
    
    print(f"Train data: {train_sequences.shape} -> {train_targets.shape}")
    print(f"Val data:   {val_sequences.shape} -> {val_targets.shape}")
    
    # Convert to tensors
    train_sequences = torch.FloatTensor(train_sequences)
    train_targets = torch.FloatTensor(train_targets)
    val_sequences = torch.FloatTensor(val_sequences)
    val_targets = torch.FloatTensor(val_targets)
    
    # Create datasets
    train_dataset = TensorDataset(train_sequences, train_targets)
    val_dataset = TensorDataset(val_sequences, val_targets)
    
    # Create data loaders
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Train single LSTM model for position prediction')
    
    # Dataset configuration
    parser.add_argument('--config', type=int, required=True, choices=[1, 2, 3],
                       help='Dataset configuration to train (1, 2, or 3)')
    
    # Paths
    parser.add_argument('--data-base-path', type=str, 
                       default='/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/utils',
                       help='Base path to data directories')
    
    parser.add_argument('--save-path', type=str, default='./models/',
                       help='Base path to save models')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=1e-5, help='Minimum improvement threshold')
    
    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Configure paths and model name based on config
    data_path = os.path.join(args.data_base_path, f"lstm_data_{args.config}")
    save_path = os.path.join(args.save_path, f"config{args.config}")
    model_name = f"LSTM_Model_Config{args.config}"
    
    print(f"="*60)
    print(f"LSTM TRAINING - CONFIGURATION {args.config}")
    print(f"="*60)
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print(f"Model name: {model_name}")
    
    # Set device and optimize for performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print(f"Using device: {device}")
    
    try:
        # Load data
        train_loader, val_loader = load_and_prepare_data(data_path, args.batch_size)
        
        # Create model
        model = PositionLSTM(
            input_size=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # Create trainer
        trainer = LSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate
        )
        
        # Train model
        best_val_loss = trainer.train(
            num_epochs=args.epochs,
            save_path=save_path,
            model_name=model_name,
            patience=args.patience,
            min_delta=args.min_delta
        )
        
        # Plot training history
        trainer.plot_training_history(save_path, model_name)
        
        print(f"\nSUCCESS! Model ready for use:")
        print(f"Model file: {os.path.join(save_path, f'{model_name}_best.pth')}")
        print(f"Final validation loss: {best_val_loss:.6f}")
        
    except Exception as e:
        print(f"\nERROR: Training failed - {str(e)}")
        raise

if __name__ == '__main__':
    main()