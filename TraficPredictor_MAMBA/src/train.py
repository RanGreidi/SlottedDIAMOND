import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from TraficPredictor_MAMBA.src.model import TrafficRNN, TrafficLSTM, TrafficGRU, TrafficMamba
import torch.nn as nn
import torch.optim as optim
from  TraficPredictor_MAMBA.src.TrafficDataset import TrafficDataset
from  TraficPredictor_MAMBA.src.simulation import history_num_slots, num_slots
from  TraficPredictor_MAMBA.src.utils import save_checkpoint



# Paths to dataset
TRAIN_DIR = "TraficPredictor_MAMBA/data/synthetic/train/"
TEST_DIR = "TraficPredictor_MAMBA/data/synthetic/test/"
CHECKPOINT_DIR = "TraficPredictor_MAMBA/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

INPUT_SEQUENCE_LENGTH = history_num_slots  # One-to-many, input length = 1
OUTPUT_SEQUENCE_LENGTH = num_slots
BATCH_SIZE = 256
EPOCHS = 10000

HIDDEN_SIZE = 256
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
        
    # Create DataLoaders for training and testing
    train_dataset = TrafficDataset(TRAIN_DIR, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)
    test_dataset = TrafficDataset(TEST_DIR, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)


    # dataset examples
    for i, (X,y) in enumerate(train_dataset):
        print(i, X.shape, y.shape)
        print(X)
        print(y)
        if i > 5: break

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # batch example
    example_batch = next(iter(train_loader))  # Get the first batch
    X_batch, y_batch = example_batch
    print("Example Batch (Before Training Starts):")
    print(f"X_batch shape: {X_batch.shape}")  # Shape of input tensor
    print(f"y_batch shape: {y_batch.shape}")  # Shape of output tensor


    # Initialize model
    model = TrafficMamba(input_size=1, hidden_size=HIDDEN_SIZE, input_sequence_length=INPUT_SEQUENCE_LENGTH, output_size=OUTPUT_SEQUENCE_LENGTH).to(DEVICE)
    criterion = nn.L1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch).to(DEVICE)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {average_loss:.4f}")
        
        # Save checkpoint after every 100 epochs
        if epoch%100 == 0:
            save_checkpoint(epoch + 1, model, optimizer, average_loss, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
        # Save the best model based on the loss
        if average_loss < best_loss and average_loss < 400:
            best_loss = average_loss
            save_checkpoint(epoch + 1, model, optimizer, average_loss, f"{CHECKPOINT_DIR}/best_model.pth")

    print("Training Complete")
