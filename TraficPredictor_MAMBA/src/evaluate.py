import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import TrafficRNN,TrafficLSTM,TrafficGRU,TrafficMamba
from torch.utils.data import Dataset, DataLoader
from train import HIDDEN_SIZE, OUTPUT_SEQUENCE_LENGTH, TEST_DIR, INPUT_SEQUENCE_LENGTH, DEVICE
from TrafficDataset import TrafficDataset
from utils import load_evaluated_checkpoint

# Load the trained model
MODEL_PATH = "TraficPredictor_MAMBA/checkpoints/best_model.pth"

# Initialize model
model =  TrafficMamba(input_size=1, hidden_size=HIDDEN_SIZE, input_sequence_length=INPUT_SEQUENCE_LENGTH, output_size=OUTPUT_SEQUENCE_LENGTH).to(DEVICE)
load_evaluated_checkpoint(model, MODEL_PATH)
model.eval()  # Set to evaluation mode

# Load test data (new Markov sequence)
test_dataset = TrafficDataset(TEST_DIR, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Run inference
all_predictions = []
all_true_values = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        # Model inference
        y_pred = model(X_batch)
        
        # Store results
        all_predictions.append(y_pred.cpu().numpy())
        all_true_values.append(y_batch.cpu().numpy())

# Convert to numpy arrays
y_true = np.concatenate(all_true_values, axis=0)
y_pred = np.concatenate(all_predictions, axis=0)


# Compute evaluation metrics
mse = ((y_true - y_pred) ** 2).mean()
mae = abs(y_true - y_pred).mean()

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# # Save actual vs. predicted plot
# plt.figure(figsize=(10, 5))
# plt.plot(y_true[:20].flatten(), label="Actual", marker="o")
# plt.plot(y_pred[:20].flatten(), label="Predicted", marker="x")
# plt.legend()
# plt.title("Traffic Flow Prediction (First 20 Samples)")
# plt.xlabel("Time Steps")
# plt.ylabel("Packet Demand")

# # Save the figure instead of showing it
# fig_path = "reports/figures/evaluation_plot.png"
# plt.savefig(fig_path)
# plt.close()

# print(f"Figure saved at: {fig_path}")

# Number of samples to visualize
num_samples = min(256, y_true.shape[0])  # Plot up to 5 samples or the available number

fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))

for i in range(num_samples):
    ax = axes[i] if num_samples > 1 else axes  # Ensure axes works for single sample
    ax.plot(y_true[i+0, :], label="Actual", marker="o", linestyle="-")
    ax.plot(np.round(y_pred[i+0, :]), label="Predicted", marker="x", linestyle="--")
    ax.legend()
    ax.set_title(f"Sample {i + 1}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Packet Demand")

# Adjust layout
plt.tight_layout()

# Save the figure instead of showing it
fig_path = "reports/figures/evaluation_multiple_samples.png"
plt.savefig(fig_path)
plt.close()

print(f"Figure saved at: {fig_path}")