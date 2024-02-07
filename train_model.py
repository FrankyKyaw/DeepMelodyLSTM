from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding
from preprocess import preprocess_notes
from keras.optimizers import Adam

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = "model.h5"

def generate_model(sequence_length, n_vocab, embedding_dim=100, dropout=0.35, learning_rate=LEARNING_RATE):
    model = Sequential()
    model.add(Embedding(n_vocab, embedding_dim, input_length=sequence_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))

    model.summary()

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

def train(sequence_length, learning_rate=LEARNING_RATE):

    n_vocab, network_input, network_output = preprocess_notes(sequence_length=sequence_length)

    model = generate_model(sequence_length, n_vocab)

    train_model(model, network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save(MODEL_PATH)


if __name__ == "__main__":
    train()