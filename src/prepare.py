# Step 1: download Fashion-MNIST and save it as .npy files in data/raw/
import os
import numpy as np
from tensorflow import keras

os.makedirs("data/raw", exist_ok=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

np.save("data/raw/x_train.npy", x_train)
np.save("data/raw/y_train.npy", y_train)
np.save("data/raw/x_test.npy", x_test)
np.save("data/raw/y_test.npy", y_test)

print("Saved raw data:", x_train.shape, x_test.shape)
