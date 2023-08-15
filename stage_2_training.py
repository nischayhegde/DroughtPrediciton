import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
california = pd.read_csv('california.csv')

def create_sequences(df, seq_length=24):
    df["Week"] = pd.to_datetime(df["Week"])
    df["Week"] = df["Week"].dt.isocalendar().week
    df=df.iloc[::-1]
    data=df.values
    data=np.concatenate(((data[:,0])[:, np.newaxis], data), axis=1)
    data=data.astype(np.float32)
    data[:,0]=np.sin(2 * np.pi * data[:,0] / 52)
    data[:,1]=np.cos(2 * np.pi * data[:,1] / 52)
    print(data.shape)
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length,2:8]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

x, y = create_sequences(california, 104)

# Assuming we have 9 features (Sinweek, Cosweek, None, D0-D4, D1-D4, D2-D4, D3-D4, D4, DSCI) and 6 targets (None, D0-D4, D1-D4, D2-D4, D3-D4, D4)
# Define the second stage neural network model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, linear_dim, output_dim):
        super(LSTMNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, linear_dim)  # Add a linear layer
        self.relu = nn.ReLU()  # Add ReLU activation
        self.fc = nn.Linear(linear_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))

        h1 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm2(out, (h1, c1))

        out = self.linear(out[:, -1, :])  # Apply the linear layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc(out)  # Connect to the final output layer
        return out
    
input_dim = 9
hidden_dim = 64
linear_dim = 32
num_layers = 2
output_dim = 6

# Load the national model
national_model = torch.load('nationwide_dp.pth')

# Instantiate the modified model
california_model = LSTMNet(input_dim, hidden_dim, num_layers, linear_dim, output_dim)

# Copy the parameters from the national model to the advanced model
california_model.lstm1.load_state_dict(national_model.lstm.state_dict())

# Freeze the parameters of the pretrained portion of the advanced model
for param in california_model.lstm1.parameters():
    param.requires_grad = False

# Instantiate the model
device = torch.device('cuda')
california_model = california_model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(california_model.parameters(), lr=0.001)

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Training loop
train_losses = []
val_losses = []
val_accuracy = []
std_errors = []
save_model=True
enum=0
try:
    for epoch in range(15000):  # Number of epochs
        california_model.train()
        optimizer.zero_grad()
        
        # Forward pass on training data
        outputs_train = california_model(X_train)
        loss_train = criterion(outputs_train, y_train)
        
        # Backpropagation and optimization
        loss_train.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Forward pass on validation data
            california_model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                outputs_val = california_model(X_val)
                loss_val = criterion(outputs_val, y_val)
                percent_error=outputs_val - y_val
                avg_percent_error = torch.mean(torch.abs(percent_error)).item()
                # Calculate the standard deviation
                std_percent_error = torch.std(percent_error).item()
            # Store values for the training graph
            train_losses.append(loss_train.item())
            val_losses.append(loss_val.item())
            val_accuracy.append(100-avg_percent_error)
            std_errors.append(std_percent_error)
            print(f'Epoch: {epoch+1}, Training Loss: {loss_train.item()}, Validation Loss: {loss_val.item()}, Avg Validation Accuracy: {100-avg_percent_error}%, SD of Residuals: {std_percent_error}%')
        enum+=1

except KeyboardInterrupt:
    print("Keyboard interrupt received. Exiting...")

finally:
    # Plot the training graph
    torch.save(california_model, 'california_dp.pth')
    print("Model saved.")
    epochs = range(0, enum, 10)

    plt.plot(epochs, val_accuracy, label='AVG Accuracy (%)')
    plt.plot(epochs, std_errors, label='SD of Residuals(%)')

    plt.xlabel('Epochs')
    plt.ylabel('Validation Split Stats (california)')
    plt.legend()
    plt.savefig('california_training_graph.png')
    plt.show()