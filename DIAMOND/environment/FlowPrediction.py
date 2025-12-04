import sys
import os

# --------------------------------------------------------------------------------------- #
# Step 1: Go up two levels from current file to reach the project root
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Step 2: Add it to sys.path
# sys.path.append(project_root)
# --------------------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from TraficPredictor_MAMBA.src.model import TrafficRNN, TrafficLSTM, TrafficGRU, TrafficMamba
from torch.utils.data import Dataset, DataLoader
from TraficPredictor_MAMBA.src.train import HIDDEN_SIZE, OUTPUT_SEQUENCE_LENGTH, TEST_DIR, INPUT_SEQUENCE_LENGTH, DEVICE
from TraficPredictor_MAMBA.src.TrafficDataset import TrafficDataset
from TraficPredictor_MAMBA.src.utils import load_evaluated_checkpoint


class FlowPrediction:
    
    def __init__(self,flow_statistics):
        
        # Input
        self.flow_history_count = flow_statistics.history_count
               
        # General
        self.pkt_arrival_sample_rate = flow_statistics.pkt_arrival_sample_rate
        self.type_scaler = flow_statistics.type_scaler        
        self.MODEL_PATH = "TraficPredictor_MAMBA/checkpoints/best_model.pth"
        self.True_future_count = flow_statistics.future_count
        self.True_future_evets = flow_statistics.future_events

        # Output
        self.flow_name = flow_statistics.flow_name
        self.predicted_count = self.predict_future_count()
        self.predicted_events = self.convert_counts2events()

        if True:
            print(f"Prediction Mean Squared Error (MSE) of flow {self.flow_name}: {self.prediction_MSE:.4f}")
            print(f"Prediction Mean Absolute Error (MAE) of flow {self.flow_name}: { self.prediction_MAE:.4f}")

    def step(self, slot):
        """
        Takes one step in the Markov Chain by transitioning to the next state based on the transition matrix.
        """
        # current_event = self.future_events[slot] * self.type_scaler
        current_event = sum(self.predicted_events[slot-self.pkt_arrival_sample_rate:slot]) * self.type_scaler
        return current_event

    def predict_future_count(self):
        
        # Initialize model
        model = TrafficMamba(input_size=1, hidden_size=HIDDEN_SIZE, input_sequence_length=INPUT_SEQUENCE_LENGTH, output_size=OUTPUT_SEQUENCE_LENGTH).to(DEVICE)
        load_evaluated_checkpoint(model, self.MODEL_PATH)
        model.eval()  # Set to evaluation mode

        # make a batch with a single input sequance
        input_sequence = np.array(self.flow_history_count, dtype=np.float32)
        input_sequence = np.expand_dims(input_sequence, axis=(0, 2))
        input_tensor = torch.from_numpy(input_sequence).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            y_pred = model(input_tensor).cpu()
        
        # post process the prediction
        y_pred = np.round(y_pred).numpy()
        y_pred = y_pred[:len(self.True_future_count)]
        
        self.prediction_MAE = abs(self.True_future_count - y_pred).mean()
        self.prediction_MSE = ((self.True_future_count - y_pred) ** 2).mean()

        return y_pred
    
    def convert_counts2events(self):
        self.predicted_count
        self.True_future_count
        self.True_future_evets

        # check that the convertion works: if self.True_future_evets == debug_on_true_values
        debug_on_true_values = [self.True_future_count[count_idx] - self.True_future_count[count_idx-1] for count_idx in range(1,len(self.True_future_count))]        
        debug_on_true_values.insert(0, self.True_future_count[0]-self.flow_history_count[-1])

        predicted_events = [self.predicted_count[count_idx] - self.predicted_count[count_idx-1] for count_idx in range(1,len(self.predicted_count))]
        predicted_events.insert(0, self.predicted_count[0]-self.flow_history_count[-1])
        return predicted_events