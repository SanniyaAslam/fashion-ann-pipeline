# Step 3: build and train the ANN
import os
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

# read settings from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

os.makedirs("models", exist_ok=True)

# load processed data
x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

# Flatten -> Dense (ReLU) -> Dropout -> Dense(10, Softmax)
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

# save the model and the training history
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Model saved to models/model.h5")
