# Step 4: test the model, save a confusion matrix picture and metrics.json
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw pictures without opening a window
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("reports", exist_ok=True)

# load test data and the trained model
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")
model = keras.models.load_model("models/model.h5")

# test loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# confusion matrix picture
predictions = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("reports/confusion_matrix.png")

# write metrics.json in the project root
with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=2)

print("Test accuracy:", accuracy)
