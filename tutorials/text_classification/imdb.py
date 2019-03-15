import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras

import text_translator

VOCAB_SIZE = 10000
MAX_LENGTH = 512

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding="post",
                                                        maxlen=MAX_LENGTH)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding="post",
                                                       maxlen=MAX_LENGTH)

translator = text_translator.TextTranslator(imdb.get_word_index())

def create_model(train_data, train_labels, validation_size=10000):
  model = keras.Sequential()
  model.add(keras.layers.Embedding(VOCAB_SIZE, 16))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(keras.layers.Dense(16, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

  model.summary()

  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

  # We take a subset of the training data to use for validation at training time.
  history = model.fit(
    train_data[validation_size:],
    train_labels[validation_size:],
    epochs=40,
    batch_size=512,
    validation_data=(train_data[:validation_size], train_labels[:validation_size]),
    verbose=1
  )
  model.save("review_classifier.h5")
  with open("review_classifier.history", "wb") as wh:
    pickle.dump(history.history, wh)
  return

def eval_model():
  model = keras.models.load_model("review_classifier.h5")
  results = model.evaluate(test_data, test_labels)
  print(results)

  with open("review_classifier.history", "rb") as rh:
    history_dict = pickle.load(rh)
  acc = history_dict['acc']
  val_acc = history_dict['val_acc']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  plt.figure(figsize=(8, 12))
  plt.subplot(2, 1, 1)
  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()

def custom_eval(text):
  model = keras.models.load_model("review_classifier.h5")
  data = keras.preprocessing.sequence.pad_sequences(
    [translator.encode_review(text)], value=0, padding="post", maxlen=MAX_LENGTH
  )
  prediction = model.predict(data)
  print(prediction[0])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Classify some text!")
  parser.add_argument("--mode", choices=["train", "eval", "custom"],
                      help="How to run the program.",
                      default="eval")
  parser.add_argument("--text", help="Custom text to evaluate by the model.")
  args = parser.parse_args()
  if args.mode == "train":
    create_model()

  elif args.mode == "eval":
    eval_model()

  elif args.mode == "custom":
    custom_eval(args.text)
