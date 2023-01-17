from keras_gnn import BGNN
from keras.layers import Input
from keras.models import Model

# Define the input layer
input_layer = Input(shape=(num_nodes, num_features))

# Define the BGNN layer
bgcn_layer = BGNN(units=32, activation='relu', num_samplings=10)(input_layer)

# Define the output layer
output_layer = Dense(num_classes, activation='softmax')(bgcn_layer)

# Create the model
model = Model(input_layer, output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
