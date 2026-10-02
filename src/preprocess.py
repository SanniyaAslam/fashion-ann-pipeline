# Step 2: normalize the images and create a validation set
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

# read settings from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

os.makedirs("data/processed", exist_ok=True)

# load raw data
x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_test = np.load("data/raw/y_test.npy")

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# split a validation set out of the training data
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
)

# save everything (float32 keeps the files smaller)
np.save("data/processed/x_train.npy", x_train.astype("float32"))
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val.astype("float32"))
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test.astype("float32"))
np.save("data/processed/y_test.npy", y_test)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)
