import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import pyro

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder



class IrisDataset(Dataset):
    def __init__(self, csv_file):
        #data = pd.read_csv(csv_file)
        iris = datasets.load_iris()
        self.X, self.y = iris.data, iris.target
        
        # Shuffle the data
        idx = torch.randperm(len(self.X))
        self.X = self.X[idx]
        self.y = self.y[idx]

        encoder = OneHotEncoder()
        self.y = encoder.fit_transform(self.y.reshape(-1, 1)).toarray()

        #self.X = data.drop(['species'], axis=1)
        #self.y = data['species']
        self.batch_size = 32
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx], self.y[idx]

    def num_batches(self):
        num_batches = len(self.X) // self.batch_size
        if len(self.X) % self.batch_size != 0:
            num_batches += 1
        return num_batches
        
    def get_batch(self, batch_id):
        start = batch_id * self.batch_size
        end = start + self.batch_size
        if end > len(self.X):
            end = len(self.X)

        return self.X[start:end], self.y[start:end]




class BayesianModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        h = torch.relu(self.hidden_layer(h))
        y = self.output_layer(h)
        return y
    
    def guide(self, x):
        for name, param in self.named_parameters():
            if "weight" in name:
                mean = param
                scale = torch.ones_like(param)
                pyro.sample(name, Normal(mean, scale).to_event(1))
            else:
                mean = param
                scale = torch.ones_like(param)
                pyro.sample(name, Normal(mean, scale).to_event(1))



print(torch.__version__)

# Define the number of training epochs
num_epochs = 200

# Load dataset 
dataset = IrisDataset('iris.csv')
dataset.batch_size = 32

print("Dataset length: ", len(dataset))


# Define the input, hidden, and output dimensions
input_dim = len(dataset.X[0])
hidden_dim = 16
output_dim = 3

print("Input dim: ", input_dim)


# Define the data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
model = BayesianModel(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


x_test, y_test = dataset.get_batch(4)

# One hot encoder to vector
y_test = torch.argmax(torch.tensor(y_test), dim=1).numpy()
print("y_test ", y_test)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
#y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#print(y_test)


num_epochs = 200
for i in range(num_epochs):
    for batch_id in range(dataset.num_batches()):
        #print("Batch: ", batch_id)
        x, y = dataset.get_batch(batch_id)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Forward pass
        output = model.forward(x_tensor)
        loss = loss_fn(output, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("Loss: ", loss.item())

        # Compute accuracy against test set
        output = model.forward(x_test_tensor)
        _, predicted = torch.max(output.data, 1)

        # Tensor to numpy
        predicted = predicted.numpy()

        #print(predicted)
        correct = (predicted == y_test).sum().item()
        total = len(y_test)
        #print("Correct: ", correct, "/", total)
        #print(correct, "/", total)
        print('Accuracy: \t%d %%' % (100 * correct / total))

        #print(x_tensor)




"""
# Train the model
for i in range(num_epochs):
    print("Epoch: ", i)
    for data in dataloader:
        x, y = data
        # Forward pass
        output = model(x)
        loss = loss_fn(output, y)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Use variational inference to fill missing values
        for name, param in model.named_parameters():
            if param.requiresGrad:
                pyro.sample(name, Normal(param, 0.1).to_event(1))
"""