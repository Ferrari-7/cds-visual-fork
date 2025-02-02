"""
This script trains a simple neural networks using TensorFlow.
TensorFlow is a machine learning and deep learning framework developed by Google Research. You can find the documentation [here](https://www.tensorflow.org/).
The script will train on greyscale images.

Please make sure that the necessary packages have been installed by navigating to main repository and running this in command line:

bash setup.sh

"""

# Importing packages
# generic tools
import numpy as np
# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
# data loader
from tensorflow.keras.datasets import cifar10

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # creating labels
    labels = ["airplane",
             "automobile",
             "bird",
             "cat",
             "deer",
             "dog",
             "frog",
             "horse",
             "ship",
             "truck"]
    # normalise data (F: making between 0 and 1)
    data = data.astype("float")/255.0
    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)
    # convert labels to one-hot encoding (F: making an array. each label has a specific array.)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return data, labels, X_train, X_test, y_train, y_test

# Defining architecture for neural network.
def network_arch():
    # define architecture 784x256x128x10 (input, first layer, second layer, output)
    model = Sequential()
    model.add(Dense(256, 
                    input_shape=(784,), 
                    activation="relu"))
    model.add(Dense(128, 
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax"))
    # train model using SGD. Defining learning rate.
    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy",
              optimizer=sgd, 
              metrics=["accuracy"])
    return model

# Fitting model to data.
def fit_data(model):
    model.fit(X_train, y_train,  epochs=10, batch_size=32)
    return model

# Creating a classification report.
def clf_report(X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), 
                            predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_]))

def main():
    load_data()
    network_arch()
    fit_data(model)
    clf_report(X_test)


if __name__=="__main__":
    main()
