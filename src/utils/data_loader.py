from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_name = "mnist"):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset name")

    #convert to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Normalize
    x_train /= 255.0
    x_test /= 255.0

    # Flatten
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # One hot encode labels
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # train validation split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test




