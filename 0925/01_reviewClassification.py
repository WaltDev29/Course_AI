import tensorflow as tensorflow
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words = 10000)