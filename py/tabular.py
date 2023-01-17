import os
import networkx as nx
import pandas as pd

data_dir = ("./cora")

edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"

print(edgelist.sample(frac=1).head(5))

"""
import numpy as np
from keras_gnn import BGNN
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load a standard graph dataset, like Cora, Citeseer or Pubmed

# create the adjacency matrix and the feature matrix
A = dataset.adj
X = dataset.features
y = dataset.labels

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the input layer
input_layer = Input(shape=(X.shape[1],))

# Define the BGNN layer
bgcn_layer = BGNN(units=32, activation='relu', num_samplings=10)(input_layer)

# Define the output layer
output_layer = Dense(y.shape[1], activation='softmax')(bgcn_layer)

# Create the model
model = Model(input_layer, output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32)

# evaluate the model on the test data
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.4f}')
"""