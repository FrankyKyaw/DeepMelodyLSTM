
from preprocess import preprocess_notes


import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, Concatenate, Input
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


sequence_length = 16
EPOCHS = 100
BATCH_SIZE = 64
learning_rate = 0.001
sequence_length = 16
note_embedding_dim, chord_embedding_dim = 100, 100
lstm_units = 256
dropout_rate = 0.35
MODEL_PATH = "model.h5"
output_path = "output.mid"

def hierarchical_lstm_model(sequence_length, n_vocab_notes, n_vocab_chords, note_embedding_dim, chord_embedding_dim, lstm_units, dropout_rate, learning_rate):

    note_input = Input(shape=(sequence_length,), name='note_input')
    note_embedding = Embedding(input_dim=n_vocab_notes + 1, output_dim=note_embedding_dim, input_length=sequence_length, name='note_embedding')(note_input)
    note_lstm = LSTM(lstm_units, return_sequences=True, name='note_lstm')(note_embedding)
    note_lstm_dropout = Dropout(dropout_rate, name='note_dropout')(note_lstm)

    chord_input = Input(shape=(sequence_length,), name='chord_input')
    chord_embedding = Embedding(input_dim=n_vocab_chords + 1, output_dim=chord_embedding_dim, input_length=sequence_length, name='chord_embedding')(chord_input)
    chord_lstm = LSTM(lstm_units, return_sequences=True, name='chord_lstm')(chord_embedding)
    chord_lstm_dropout = Dropout(dropout_rate, name='chord_dropout')(chord_lstm)

    combined = Concatenate(name='concatenate')([note_lstm_dropout, chord_lstm_dropout])
    combined_lstm = LSTM(lstm_units, name='combined_lstm')(combined)
    combined_dropout = Dropout(dropout_rate, name='combined_dropout')(combined_lstm)
    
    chord_output = Dense(n_vocab_chords + 1, activation='softmax', name='chord_output')(combined_dropout)
    note_output = Dense(n_vocab_notes + 1, activation='softmax', name='note_output')(combined_dropout)
    
    model = Model(inputs=[note_input, chord_input], outputs=[note_output, chord_output])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss={'note_output': 'sparse_categorical_crossentropy', 'chord_output': 'sparse_categorical_crossentropy'},
                  loss_weights={'note_output': 1.0, 'chord_output': 1.0})

    return model

def train_model(model, network_input, network_output, epochs=200,
                batch_size=64):
    """
    Train the model with the given number of epochs and batch size and
    saves the weights in the filepath every 20 epochs.

    Args:
        model (_type_): _description_
        network_input (_type_): _description_
        network_output (_type_): _description_
        epochs (int, optional): _description_. Defaults to 200.
        batch_size (int, optional): _description_. Defaults to 64.
        filepath (str, optional): _description_. Defaults to 'weights-{epoch:02d}-{loss:.4f}.hdf5'.

    Returns:
        _type_: _description_
    """

    model.fit(network_input, network_output, epochs=epochs,
              batch_size=batch_size)

def train():
    network_input = [network_input_notes, network_input_chords]
    network_output = [network_output_notes, network_output_chords]

    model = hierarchical_lstm_model(sequence_length, n_vocab_notes, n_vocab_chords, note_embedding_dim, chord_embedding_dim, lstm_units, dropout_rate, learning_rate)

    train_model(model, network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save(MODEL_PATH)


if __name__ == "__main__":
    train()