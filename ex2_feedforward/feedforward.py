# Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Create a feedforward neural network model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model using the categorical cross-entropy loss function and the Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the input and output data, for a fixed number of epochs
model.fit(X_train, np.eye(3)[y_train], epochs=100, batch_size=10, verbose=0)

# Evaluate the model using the testing set
loss, accuracy = model.evaluate(X_test, np.eye(3)[y_test])
print('Accuracy:', accuracy)