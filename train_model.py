from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding
import math
from keras.callbacks import ModelCheckpoint

def generate_model(sequence_length, n_vocab):
    model = Sequential()
    model.add(Embedding(n_vocab, 100, input_length=sequence_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.35))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.35))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.35))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model

def train_model(model, network_input, network_output, epochs=200, 
                batch_size=64, filepath='weights-{epoch:02d}-{loss:.4f}.hdf5'):
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
    n_batches = len(network_input) / batch_size
    n_batches = math.ceil(n_batches)
    checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_inly=True,
    mode='min',
    save_freq=20*n_batches
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=epochs, 
              batch_size=batch_size, callbacks=callbacks_list)
    