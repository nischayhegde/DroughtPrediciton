import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
regions=[pd.read_csv('midwest.csv'), pd.read_csv('northeast.csv'), pd.read_csv('northernforests.csv'), pd.read_csv('northernplains.csv'), pd.read_csv('northwest.csv'), pd.read_csv('southeast.csv'), pd.read_csv('southernplains.csv'), pd.read_csv('southwest.csv')]

def create_sequences(df, seq_length=24):
    df["Week"] = pd.to_datetime(df["Week"])
    df["Week"] = df["Week"].dt.isocalendar().week
    df=df.iloc[::-1]
    data=df.values
    data=np.concatenate(((data[:,0])[:, np.newaxis], data), axis=1)
    data=data.astype(np.float32)
    data[:,0]=np.sin(2 * np.pi * data[:,0] / 52)
    data[:,1]=np.cos(2 * np.pi * data[:,1] / 52)
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length,2:8]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

xlist=[]
ylist=[]
for region in regions:
    x,y=create_sequences(region, 104)
    xlist.append(x)
    ylist.append(y)

x = np.concatenate(tuple(xlist), axis=0)
y = np.concatenate(tuple(ylist), axis=0)
print(x.shape)
print(y.shape)

# Assuming we have 9 features (Sinweek, Cosweek, None, D0-D4, D1-D4, D2-D4, D3-D4, D4, DSCI) and 6 targets (None, D0-D4, D1-D4, D2-D4, D3-D4, D4)
# Define the first stage neural network model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
    
input_dim = 9
hidden_dim = 64
num_layers = 2
output_dim = 6

# Instantiate the model
device = torch.device('cuda')
model = LSTMNet(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

batch_size = 2048
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
train_losses = []
val_losses = []
val_accuracy = []
std_errors = []
save_model=True
enum=0
try:
    for epoch in range(3000):  # Number of epochs
        model.train()
        optimizer.zero_grad()
        
        for batch_X, batch_y in train_loader:
            # Forward pass on training data
            outputs_train = model(batch_X)
            loss_train = criterion(outputs_train, batch_y)

            # Backpropagation and optimization
            loss_train.backward()
            optimizer.step()

        if epoch % 10 == 0:
            # Forward pass on validation data
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                outputs_val = model(X_val)
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
    if save_model:
        torch.save(model, 'nationwide_dp.pth')
        print("Model saved.")
    epochs = range(0, enum, 10)

    plt.plot(epochs, val_accuracy, label='Mean Accuracy (%)')
    plt.plot(epochs, std_errors, label='SD of Residuals(%)')

    plt.xlabel('Epochs')
    plt.ylabel('Validation Split Stats (nationwide)')
    plt.legend()
    plt.savefig('nationwide_training_graph.png')
    plt.show()