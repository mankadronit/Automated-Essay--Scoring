from constants import GLOVE_DIR
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
from utils import tokenizer, load_embedding_matrix

def get_model(embedding_dimension, essay_length):
    vocabulary_size = len(tokenizer.word_index) + 1
    embedding_matrix = load_embedding_matrix(glove_directory=GLOVE_DIR, embedding_dimension=embedding_dimension)

    model = Sequential()

    model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=False, mask_zero=False))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(256, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.summary()

    return model