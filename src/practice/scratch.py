import torch
import torch.nn as nn

class LSTMSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        output = self.fc(lstm_out)  # output shape: (batch_size, sequence_length, output_size)
        return output

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    num_layers = 2
    output_size = 5
    sequence_length = 15
    batch_size = 8

    # Create the model
    model = LSTMSequenceModel(input_size, hidden_size, num_layers, output_size)
    
    # Create a random input tensor (batch_size, sequence_length, input_size)
    input_tensor = torch.randn(batch_size, sequence_length, input_size)
    
    # Forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (batch_size, sequence_length, output_size)
