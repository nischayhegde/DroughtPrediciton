import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
# Load the california model
model = torch.load('california_dp.pth')

# Instantiate the model
device = torch.device('cuda')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

california = pd.read_csv('california.csv')

def create_sequence(df, seq_length=24):
    df["Week"] = pd.to_datetime(df["Week"])
    df["Week"] = df["Week"].dt.isocalendar().week
    df=df.iloc[::-1]
    data=df.values
    data=np.concatenate(((data[:,0])[:, np.newaxis], data), axis=1)
    data=np.concatenate(((data[:,0])[:, np.newaxis], data), axis=1)
    data=data.astype(np.float32)
    data[:,1]=np.sin(2 * np.pi * data[:,1] / 52)
    data[:,2]=np.cos(2 * np.pi * data[:,2] / 52)
    return np.array(data[-seq_length:])

forecasts=[]

def autoregressive(data, seq_length=24, index=1, stop=12):
    input=torch.tensor(np.expand_dims(data[:,1:], axis=0), dtype=torch.float32).to(device)
    forecast=model(input).cpu().detach().numpy()
    forecasts.append(forecast)
    week=data[seq_length-1,0]+1
    last_row=np.concatenate((np.expand_dims(np.squeeze(np.array([week, np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52)])), 0), forecast), axis=1)
    last_row=np.concatenate((last_row, np.expand_dims(np.expand_dims(np.array(np.sum(forecast[0,1:])), axis=0), axis=0)), axis=1)
    data=np.concatenate((data[1:seq_length, :], last_row), axis=0)
    if index<stop:
        autoregressive(data, seq_length, index+1, stop)

data=create_sequence(california, 104)
autoregressive(data, 104, 1, 52)
graph1data=np.squeeze(np.abs(np.array(forecasts)))

severity_labels = ['No Drought', 'D0+', 'D1+', 'D2+', 'D3+', 'D4+']

# Create a list of colors for your severities
severity_colors = ['white', 'yellow', (1, 0.6, 0), (1, 0.4, 0), 'red', 'maroon']

plt.figure(figsize=(15, 5))

# Create a bar for each severity at each timestep
for j in range(graph1data.shape[1]):  # start from the last column to have the highest severity in front
    plt.bar(range(graph1data.shape[0]), graph1data[:, j], width=1, color=severity_colors[j], label=severity_labels[j])

plt.title('California Drought Severity Forecast Summary')
plt.xlabel('Upcoming Weeks')
plt.ylabel('Area of Drought Severity (%)')
plt.legend()
plt.savefig('california_drought_forecast.png')
plt.show()

def margin(initialsd, z, week):
    return z*math.sqrt(week*initialsd**2)

# Calculate the margin of error of the forecast
def normalspread(sd, z, severitydata):
    nodrought = np.squeeze(severitydata[:, 0])
    z_above = []
    z_below = []
    for i in range(nodrought.shape[0]):
        z_above.append(min(nodrought[i]+margin(sd, z, i+1), 100))
        z_below.append(max(nodrought[i]-margin(sd, z, i+1), 0))
    return np.array(z_above), nodrought, np.array(z_below)

above, middle, below = normalspread(3.31, 1.645, graph1data)

plt.figure(figsize=(15, 5))

# Plot the lines
plt.plot(above, label='Real Value 96th Percentile', color='red')
plt.plot(middle, label='Drought-Free Prediction', color='blue')
plt.plot(below, label='Real Value 4th Percentile', color='purple')

# Fill the areas between the lines
plt.fill_between(range(len(middle)), above, middle, color='blue', alpha=0.5)
plt.fill_between(range(len(middle)), middle, below, color='blue', alpha=0.5)

plt.title('Drought-Free Percent Area Prediction Spread')
plt.xlabel('Upcoming Weeks')
plt.ylabel('Drought-Free Land (%)')
plt.legend()
plt.savefig('df_normal_spread.png')
plt.show()


