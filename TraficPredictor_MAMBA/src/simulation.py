import os
import numpy as np
import pandas as pd
from  TraficPredictor_MAMBA.src.Traffic_Probability_HawkesModel import HawkesModel

# Hawkes Params
lambda0 = 0.005
alpha = 0.025
beta = 0.0001
num_slots=50
history_num_slots=200

# Configuration
NUM_TRAIN_REALIZATIONS = 150000  # Number of train samples
NUM_TEST_REALIZATIONS = 256   # Number of test samples
NUM_SAMPLES_TRAIN = num_slots+history_num_slots    # Length of each realization in train
NUM_SAMPLES_TEST = num_slots+history_num_slots       # Length of each realization in test

# Directories
TRAIN_DIR = "TraficPredictor_MAMBA/data/synthetic/train/"
TEST_DIR = "TraficPredictor_MAMBA/data/synthetic/test/"

# Ensure directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def generate_markov_traffic(num_samples: int, Traffic_Model: HawkesModel) -> pd.DataFrame:
    """
    Generate synthetic packet arrivals using a Markov process.

    Args:
        num_samples (int): Number of time steps to simulate.
        Traffic_Model (Traffic_Probability_Model): Markov model instance.

    Returns:
        pd.DataFrame: Generated time-series data with 'time' and 'packets' columns.
    """
    packet_counts = np.append(Traffic_Model.history_count,Traffic_Model.future_count)
    return pd.DataFrame({'time': np.arange(num_samples), 'packets': packet_counts})

def save_traffic_data(df: pd.DataFrame, filepath: str):
    """
    Save the generated traffic data to a CSV file.

    Args:
        df (pd.DataFrame): Traffic data.
        filepath (str): Path to save the CSV file.
    """
    df.to_csv(filepath, index=False, header=False)  # No headers to match previous dataset

if __name__ == "__main__":
    
    # Generate and save train realizations
    for i in range(NUM_TRAIN_REALIZATIONS):
        Traffic_Model = HawkesModel(lambda0=lambda0, alpha=alpha, beta=beta, source=None, destination=None, flow_name=i, num_slots=num_slots, history_num_slots=history_num_slots, seed=123 + i)
        traffic_data = generate_markov_traffic(NUM_SAMPLES_TRAIN, Traffic_Model)
        save_traffic_data(traffic_data, os.path.join(TRAIN_DIR, f"realization_{i}.csv"))
        print(f"sample {i} generated")
    # Generate and save test realizations
    for i in range(NUM_TEST_REALIZATIONS):
        Traffic_Model = HawkesModel(lambda0=lambda0, alpha=alpha, beta=beta, source=None, destination=None, flow_name=i + NUM_TRAIN_REALIZATIONS, num_slots=num_slots, history_num_slots=history_num_slots, seed=456 + i)
        traffic_data = generate_markov_traffic(NUM_SAMPLES_TEST, Traffic_Model)
        save_traffic_data(traffic_data, os.path.join(TEST_DIR, f"realization_{i}.csv"))

    print(f"Generated {NUM_TRAIN_REALIZATIONS} training and {NUM_TEST_REALIZATIONS} testing realizations.")
