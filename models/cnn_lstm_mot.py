from constants import GLOVE_DIR
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
from layers import Conv1DWithMasking
import keras.regularizers
import keras.backend as K
from utils import tokenizer, load_embedding_matrix

def get_model(embedding_dimension, essay_length):
    """
    Returns compiled model.
    """
    vocabulary_size = len(tokenizer.word_index) + 1
    embedding_matrix = load_embedding_matrix(glove_directory=GLOVE_DIR, embedding_dimension=embedding_dimension)
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=True, mask_zero=True))
    model.add(Conv1DWithMasking(nb_filter=64, filter_length=3, border_mode='same', subsample_length=1))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=True)))
    model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.l2(0.0)))
    model.compile(loss='mse', optimizer='rmsprop')

    return model