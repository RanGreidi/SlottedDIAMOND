import torch
import torch.nn as nn
from mamba_ssm import Mamba # pip install mamba-ssm==1.2.0.post1

class TrafficLSTM(nn.Module):
    """
    One-to-Many LSTM model for traffic demand prediction.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features per time step (usually 1 for packet count).
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of LSTM layers.
            prediction_length (int): Number of future time steps to predict.
        """
        super(TrafficLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input sequence of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Predicted packet demand for the next `prediction_length` steps.
        """
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last LSTM output (batch_size, hidden_size)
        output = self.fc(lstm_out)  # (batch_size, prediction_length)
        return output
    
class TrafficRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TrafficRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Simple RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer for Output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)

        # Use the last time step output
        out = self.fc(out[:, -1, :])
        return out

class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TrafficGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Simple RNN Layer
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer for Output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)

        # Use the last time step output
        out = self.fc(out[:, -1, :])
        # out = torch.round(out)
        return out

class TrafficMamba(nn.Module):
    def __init__(self, input_size, hidden_size, input_sequence_length, output_size):
        super(TrafficMamba, self).__init__()
        self.hidden_size = hidden_size

        # Simple RNN Layer
        self.mamba = Mamba(
                            # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=1, # Model dimension d_model
                            d_state=hidden_size,  # SSM state expansion factor
                            d_conv=4,    # Local convolution width
                            expand=2,    # Block expansion factor
                        )
        # Fully Connected Layer for Output
        self.fc = nn.Linear(input_sequence_length, output_size)

    def forward(self, x):
        
        # RNN forward pass
        out = self.mamba(x)

        # sqeeuze
        out = torch.squeeze(out)

        # Use the last time step output
        out = self.fc(out)

        return out

if __name__ == "__main__":
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape
