import torch
import os
import glob
import pandas as pd
from torch.utils.data import Dataset
import re

class TrafficDataset(Dataset):
    def __init__(self, data_dir, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH):
        self.data_dir = data_dir
        self.INPUT_SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH
        self.OUTPUT_SEQUENCE_LENGTH = OUTPUT_SEQUENCE_LENGTH
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files  # Dataset length is number of files

    def __getitem__(self, idx):
        # Load a single CSV file per batch
        file_path = self.files[idx]
        df = pd.read_csv(file_path, header=None)  # No headers
        packets = df.iloc[:, 1].values  # Extract only 'packets' column

        # Split the data into X (input sequence) and y (output sequence)
        X = packets[:self.INPUT_SEQUENCE_LENGTH]  # First INPUT_SEQUENCE_LENGTH packets as input sequence
        y = packets[self.INPUT_SEQUENCE_LENGTH:self.INPUT_SEQUENCE_LENGTH + self.OUTPUT_SEQUENCE_LENGTH]  # Next OUTPUT_SEQUENCE_LENGTH packets as output sequence

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).view(self.INPUT_SEQUENCE_LENGTH, 1)  # (INPUT_SEQUENCE_LENGTH,)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(self.OUTPUT_SEQUENCE_LENGTH)  # (OUTPUT_SEQUENCE_LENGTH,)


        return X_tensor, y_tensor