"""
Implements LSTM with Mean over Time layer.
"""

from constants import GLOVE_DIR
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, Lambda
from keras.models import Sequential
import keras.regularizers
from utils import tokenizer, load_embedding_matrix
import keras.backend as K
def get_model(embedding_dimension, essay_length):
    """
    Returns compiled model.
    """
    vocabulary_size = len(tokenizer.word_index) + 1
    embedding_matrix = load_embedding_matrix(GLOVE_DIR, embedding_dimension)

    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=False, mask_zero=False))
    model.add(Conv1D(filters=50, kernel_size=5, padding='same'))
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.l2(0.0)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
