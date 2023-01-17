import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample

class BGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(BGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lin1 = PyroModule[nn.Linear](in_dim, hidden_dim)
        self.lin2 = PyroModule[nn.Linear](hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y=None):
        pyro.module("bg_nn", self)
        with pyro.plate("data", x.shape[0]):
            x = x.reshape(-1, self.in_dim)
            h1 = torch.relu(self.lin1(x))
            h2 = self.lin2(h1)
            y_hat = self.softmax(h2)
            y = pyro.sample("y", dist.Categorical(y_hat), obs=y)
            return y_hat

# Define the BGNN model
in_dim = 4
hidden_dim = 8
out_dim = 3
model = BGNN(in_dim, hidden_dim, out_dim)

# Define the guide function for variational inference
guide = AutoDiagonalNormal(model)
# guide = AutoDiagonalNormal(model)

# Define the optimizer and the loss function
optimizer = Adam({"lr": 0.01})
svi = SVI(model.forward, guide, optimizer, loss=Trace_ELBO())

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
for epoch in range(100):
    svi.step(X_train, y_train)

# Use the model to make predictions on the test set
y_pred = model(X_test)

# Compute the accuracy of the predictions
acc = accuracy_score(y_test, y_pred.argmax(axis=1))
print('Test accuracy:', acc)
