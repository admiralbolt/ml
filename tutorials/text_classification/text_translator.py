class TextTranslator(object):

  def __init__(self, word_index):

    # The first indices are reserved
    self.word_index = {k:(v + 3) for k, v in word_index.items()}
    self.word_index["<PAD>"] = 0
    self.word_index["<START>"] = 1
    self.word_index["<UNKNOWN>"] = 2
    self.word_index["<UNUSED>"] = 3

    self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

  def decode_review(self, data_array):
    """Translates from a data array -> text."""
    return " ".join([self.reverse_word_index.get(i, "?") for i in data_array])

  def encode_review(self, text):
    """Translates from text -> data array."""
    encoded_review = [1]
    for word in text.lower().split():
      encoded_review.append(self.word_index.get(word, 2))
    return encoded_review
