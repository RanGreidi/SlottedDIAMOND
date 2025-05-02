import torch
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load synthetic traffic data from a CSV file.

    Args:
        filepath (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded time-series data.
    """
    return pd.read_csv(filepath)

def create_sequences(data: np.ndarray, sequence_length: int, prediction_length: int) -> tuple:
    """
    Convert time-series data into input-output sequences for training.

    Args:
        data (np.ndarray): Time-series data (packet counts).
        sequence_length (int): Length of the input sequence.
        prediction_length (int): Number of future steps to predict.

    Returns:
        tuple: (X, y) where:
            - X is a NumPy array of input sequences (shape: [num_samples, sequence_length, 1]).
            - y is a NumPy array of output sequences (shape: [num_samples, prediction_length]).
    """
    X, y = [], []

    for i in range(len(data) - sequence_length - prediction_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length : i + sequence_length + prediction_length])

    return np.array(X).reshape(-1, sequence_length, 1), np.array(y)

def convert_to_tensors(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Convert NumPy arrays to PyTorch tensors.

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Output sequences.

    Returns:
        tuple: (X_tensor, y_tensor) as PyTorch tensors.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

if __name__ == "__main__":
    # Example usage
    filepath = "data/synthetic/train_traffic.csv"
    sequence_length = 10  # Number of past time steps to use as input
    prediction_length = 5  # Number of future steps to predict

    # Load and preprocess data
    df = load_data(filepath)
    X, y = create_sequences(df["packets"].values, sequence_length, prediction_length)
    X_tensor, y_tensor = convert_to_tensors(X, y)

    print(f"Input Shape: {X_tensor.shape}")  # (num_samples, sequence_length, 1)
    print(f"Output Shape: {y_tensor.shape}")  # (num_samples, prediction_length)
