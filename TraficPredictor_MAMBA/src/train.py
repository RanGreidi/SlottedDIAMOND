import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from TraficPredictor_MAMBA.src.model import TrafficRNN, TrafficLSTM, TrafficGRU  # , TrafficMamba
import torch.nn as nn
import torch.optim as optim
from  TraficPredictor_MAMBA.src.TrafficDataset import TrafficDataset
from  TraficPredictor_MAMBA.src.simulation import history_num_slots, num_slots
from  TraficPredictor_MAMBA.src.utils import save_checkpoint
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split


# Paths to dataset
TRAIN_DIR = r'C:\Users\beaviv\Datasets\Hawkws_Model_Data\synthetic\train'  # "TraficPredictor_MAMBA/data/synthetic/train/"
TEST_DIR = r'C:\Users\beaviv\Datasets\Hawkws_Model_Data\synthetic\test'  #  "TraficPredictor_MAMBA/data/synthetic/test/"
CHECKPOINT_DIR = r'C:\Users\beaviv\Datasets\Hawkws_Model_Data\models'  # "TraficPredictor_MAMBA/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

INPUT_SEQUENCE_LENGTH = history_num_slots  # One-to-many, input length = 1
OUTPUT_SEQUENCE_LENGTH = num_slots
BATCH_SIZE = 256
EPOCHS = 10000

HIDDEN_SIZE = 256
LEARNING_RATE = 0.001
WANDB_TRACKING = True
model_name = "LSTM"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    if WANDB_TRACKING:
        wandb.finish()

        wandb.login(key="3ec39d34b7882297a057fdc2126cd037352175a4")

        # Generate a unique timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize the WandB run
        wandb.init(
            project="Hawkes_Mamba",
            name=f"{model_name}_{timestamp}",  # Add timestamp to the run name
            config={

                "epochs": EPOCHS,

            }
        )

    # Create DataLoaders for training and testing
    train_dataset = TrafficDataset(TRAIN_DIR, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)

    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.3, shuffle=True)

    test_dataset = TrafficDataset(TEST_DIR, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)


    # dataset examples
    # for i, (X,y) in enumerate(train_dataset):
    #     print(i, X.shape, y.shape)
    #     print(X)
    #     print(y)
    #     if i > 5: break

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # batch example
    example_batch = next(iter(train_loader))  # Get the first batch
    X_batch, y_batch = example_batch
    print("Example Batch (Before Training Starts):")
    print(f"X_batch shape: {X_batch.shape}")  # Shape of input tensor
    print(f"y_batch shape: {y_batch.shape}")  # Shape of output tensor


    # Initialize model

    if model_name == "Mamba":
        # model = TrafficMamba(input_size=1, hidden_size=HIDDEN_SIZE, input_sequence_length=INPUT_SEQUENCE_LENGTH, output_size=OUTPUT_SEQUENCE_LENGTH).to(DEVICE)
        pass

    elif model_name == "LSTM":
        model = TrafficLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=2, output_size=OUTPUT_SEQUENCE_LENGTH).to(DEVICE)

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

        # --------------------------- Validation ---------------------------- 3
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch).to(DEVICE)

                val_loss += loss.item()

            average_valid_loss = total_loss / len(train_loader)


        if WANDB_TRACKING:
            wandb.log({
                "train_loss": average_loss,
                "validation_loss": average_valid_loss,

            }, step=epoch)


        # Save checkpoint after every 100 epochs
        # if epoch % 100 == 0:
        #     save_checkpoint(epoch + 1, model, optimizer, average_loss, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
        # Save the best model based on the loss
        if average_loss < best_loss and average_loss < 400:
            best_loss = average_loss
            save_checkpoint(epoch + 1, model, optimizer, average_loss, os.path.join(CHECKPOINT_DIR, f'{model_name}_best_model.pk'))

    print("Training Complete")
