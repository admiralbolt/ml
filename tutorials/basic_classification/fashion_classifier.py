import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import numpy as np

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def create_model(train_images, train_labels, epochs=5):
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  model.fit(train_images, train_labels, epochs=epochs)
  model.save("fashion_classifier.h5")
  return

def plot_image(image, predictions, true_label):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(image, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions)
  color = "red"
  if predicted_label == true_label:
    color = "blue"

  plt.xlabel(
    f"{class_names[predicted_label]} {int(100 * np.max(predictions))} (actual: {class_names[true_label]})",
    color = color
  )

def plot_predictions(predictions, true_label):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  prediction_plot = plt.bar(range(10), predictions, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions)

  prediction_plot[predicted_label].set_color("red")
  prediction_plot[true_label].set_color("blue")

def plot_first_n(n):
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
  train_images = train_images / 255.0
  test_images = test_images / 255.0
  mnist_model = keras.models.load_model("fashion_classifier.h5")
  predictions = mnist_model.predict(test_images)

  plot_size = int(math.sqrt(n)) + 1
  plt.figure(figsize=(4 * plot_size, 2 * plot_size))
  for i in range(n):
    plt.subplot(plot_size, plot_size * 2, i * 2 + 1)
    plot_image(test_images[i], predictions[i], test_labels[i])
    plt.subplot(plot_size, plot_size * 2, i * 2 + 2)
    plot_predictions(predictions[i], test_labels[i])
  plt.show()






if __name__ == "__main__":
  plot_first_n(30)
  # fashion_mnist = keras.datasets.fashion_mnist
  # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  # train_images = train_images / 255.0
  # create_model(train_images, train_labels)
