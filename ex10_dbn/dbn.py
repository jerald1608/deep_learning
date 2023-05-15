# Import packages
from sklearn.model_selection import train_test_split
from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score

# Loading dataset
digits = pd.read_csv("train.csv")
from sklearn.preprocessing import standardscaler
X = np.array(digits.drop(["label"], axis=1))
Y = np.array(digits["label"])

# Data scaling
ss=standardscaler()
X = ss.fit_transform(X)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
y_pred = classifier.predict(x_test)
print('nAccuracy of Prediction: %f' % accuracy_score(x_test, y_pred))
