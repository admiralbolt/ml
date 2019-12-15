# https://www.tensorflow.org/tutorials/keras/regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

dataset_path = keras.utils.get_file("auto_mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

# Drop unknown values -- in this case about 6 horsepower values are missing.
dataset = dataset.dropna()

# The origin column is categorical, so we convert to a one-hot aka separate
# binary columns.
dataset["Origin"] = dataset["Origin"].map(lambda x: {
  1: "USA",
  2: "Europe",
  3: "Japan"
}.get(x))

# get_dummies converts categorical data into columns.
dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")

# Split the data into train & test.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# s = sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

# I still don't super get why we normalize the test dataset with the train
# statistics but, whatever. From the docs:
#
# Although we intentionally generate these statistics from only the training
# dataset, these statistics will also be used to normalize the test dataset.
# We need to do that to project the test dataset into the same distribution
# that the model has been trained on.

def norm(x):
  return (x - train_stats["mean"]) / train_stats["std"]

n_train_data = norm(train_dataset)
n_test_data = norm(test_dataset)


# Now that our data is sparkly clean, let's do some actually ML.
def build_model():
  model = keras.Sequential([
    # A standard densely connected neural network layer. It has inputs for each
    # of our labels -- in this case that's 7, (Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin).
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
  return model

model = build_model()

example_batch = n_train_data[:10]
example_result = model.predict(example_batch)

# Train the model for some time!
# The patience parameter controls the amount of epochs to check for improvement
# before quitting out.
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(n_train_data, train_labels, epochs=1000, validation_split=0.2,
  verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({"Basic": history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel("MAE [MPG]")

loss, mae, mse = model.evaluate(n_test_data, test_labels, verbose=2)
print(f"Testing set Mean Abs Error: {mae:5.2f} MPG")

# Make some predictions!!!
test_predictions = model.predict(n_test_data).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)


# Error distribution.
plt.clf()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()
