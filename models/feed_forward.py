from constants import GLOVE_DIR
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
import keras.backend as K
from utils import tokenizer, load_embedding_matrix

def get_model(embedding_dimension, essay_length):
    vocabulary_size = len(tokenizer.word_index) + 1
    embedding_matrix = load_embedding_matrix(glove_directory=GLOVE_DIR, embedding_dimension=embedding_dimension)

    model = Sequential()

    model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=False, mask_zero=False))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model