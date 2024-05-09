import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM, Embedding, Concatenate, Input, Bidirectional
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def preprocess_sequences(notes, note_durations, chords, chord_durations, sequence_length, notes_mapping, chords_mapping, note_duration_mapping, chord_duration_mapping):
    
    encoded_notes = convert_to_int(notes, notes_mapping)
    encoded_note_durations = convert_to_int(note_durations, note_duration_mapping)
    encoded_chords = convert_to_int(chords, chords_mapping)
    encoded_chord_durations = convert_to_int(chord_durations, chord_duration_mapping)
    
    input_notes, input_note_durations, input_chords, input_chord_durations, output_notes, output_note_durations, output_chords, output_chord_durations = [], [], [], [], [], [], [], []
    
    for i in range(len(encoded_notes) - sequence_length):
        input_notes.append(encoded_notes[i:i+sequence_length])
        input_note_durations.append(encoded_note_durations[i:i+sequence_length])
        input_chords.append(encoded_chords[i:i+sequence_length])
        input_chord_durations.append(encoded_chord_durations[i:i+sequence_length])
        
        output_notes.append(encoded_notes[i+sequence_length])
        output_note_durations.append(encoded_note_durations[i+sequence_length])
        output_chords.append(encoded_chords[i+sequence_length])
        output_chord_durations.append(encoded_chord_durations[i+sequence_length])
        
    n_vocab_notes = len(notes_mapping)
    n_vocab_durations = len(note_duration_mapping)
    n_vocab_chords = len(chords_mapping)
    n_vocab_chord_durations = len(chord_duration_mapping)

    return np.array(input_notes), np.array(input_note_durations), np.array(input_chords), np.array(input_chord_durations), np.array(output_notes), np.array(output_note_durations), np.array(output_chords), np.array(output_chord_durations), n_vocab_notes, n_vocab_durations, n_vocab_chords, n_vocab_chord_durations

def hierarchical_lstm_model(sequence_length, n_vocab_notes, n_vocab_chords, n_vocab_melody_durations, n_vocab_chord_durations, note_embedding_dim, chord_embedding_dim, melody_duration_embedding_dim, chord_duration_embedding_dim, lstm_units, dropout_rate, learning_rate):
    note_input = Input(shape=(sequence_length,), name='note_input')
    note_embedding = Embedding(input_dim=n_vocab_notes + 1, output_dim=note_embedding_dim, input_length=sequence_length, name='note_embedding')(note_input)

    melody_duration_input = Input(shape=(sequence_length,), name='melody_duration_input')
    melody_duration_embedding = Embedding(input_dim=n_vocab_melody_durations + 1, output_dim=melody_duration_embedding_dim, input_length=sequence_length, name='melody_duration_embedding')(melody_duration_input)

    chord_input = Input(shape=(sequence_length,), name='chord_input')
    chord_embedding = Embedding(input_dim=n_vocab_chords + 1, output_dim=chord_embedding_dim, input_length=sequence_length, name='chord_embedding')(chord_input)

    chord_duration_input = Input(shape=(sequence_length,), name='chord_duration_input')
    chord_duration_embedding = Embedding(input_dim=n_vocab_chord_durations + 1, output_dim=chord_duration_embedding_dim, input_length=sequence_length, name='chord_duration_embedding')(chord_duration_input)

    # Concatenate embeddings
    note_melody_concat = Concatenate(name='note_melody_concat')([note_embedding, melody_duration_embedding])
    note_melody_lstm = LSTM(lstm_units, return_sequences=True, name='note_melody_lstm')(note_melody_concat)
    note_melody_dropout = Dropout(dropout_rate, name='note_melody_dropout')(note_melody_lstm)

    chord_chord_duration_concat = Concatenate(name='chord_duration_concat')([chord_embedding, chord_duration_embedding])
    chord_lstm = LSTM(lstm_units, return_sequences=True, name='chord_lstm')(chord_chord_duration_concat)
    chord_dropout = Dropout(dropout_rate, name='chord_dropout')(chord_lstm)

    # Combine LSTM outputs
    combined = Concatenate(name='concatenate')([note_melody_dropout, chord_dropout])
    combined_lstm = Bidirectional(LSTM(lstm_units, name='combined_lstm'))(combined)
    combined_dropout = Dropout(dropout_rate, name='combined_dropout')(combined_lstm)

    # Output layers
    note_output = Dense(n_vocab_notes + 1, activation='softmax', name='note_output')(combined_dropout)
    melody_duration_output = Dense(n_vocab_melody_durations + 1, activation='softmax', name='melody_duration_output')(combined_dropout)
    chord_output = Dense(n_vocab_chords + 1, activation='softmax', name='chord_output')(combined_dropout)
    chord_duration_output = Dense(n_vocab_chord_durations + 1, activation='softmax', name='chord_duration_output')(combined_dropout)

    model = Model(inputs=[note_input, melody_duration_input, chord_input, chord_duration_input], outputs=[note_output, melody_duration_output, chord_output, chord_duration_output])
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
                  loss={'note_output': 'sparse_categorical_crossentropy',
                        'melody_duration_output': 'sparse_categorical_crossentropy',
                        'chord_output': 'sparse_categorical_crossentropy',
                        'chord_duration_output': 'sparse_categorical_crossentropy'},
                  loss_weights={'note_output': 1.0, 'melody_duration_output': 1.0, 'chord_output': 1.0, 'chord_duration_output': 1.0})

    return model


EPOCHS = 200
BATCH_SIZE = 128
learning_rate = 0.0005
sequence_length = 64
note_embedding_dim = 150
chord_embedding_dim = 300
melody_duration_embedding_dim = 50
chord_duration_embedding_dim = 50
lstm_units = 256
dropout_rate = 0.4
MODEL_PATH = "mappings_model/choral-model.h5"
output_path = "mappings_model/choral-output.mid"


def train(continue_training=False):
    network_input = [input_notes, input_note_durations, input_chords, input_chord_durations]
    network_output = [output_notes, output_note_durations, output_chords, output_chord_durations]
    
    if continue_training:
        model = load_model(MODEL_PATH)
    else:
        model = hierarchical_lstm_model(sequence_length, n_vocab_notes, n_vocab_chords, 
                                        n_vocab_durations, n_vocab_chord_durations, 
                                        note_embedding_dim, chord_embedding_dim, 
                                        melody_duration_embedding_dim, chord_duration_embedding_dim, 
                                        lstm_units, dropout_rate, learning_rate)
    
    early_stopping = EarlyStopping(
        monitor='loss', 
        patience=10, 
        verbose=1,  
        mode='min'  
    )

    model.fit(network_input, network_output,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[early_stopping])
    
    model.save(MODEL_PATH)

if __name__ == "__main__":
    sequence_length = 64
    input_notes, input_note_durations, input_chords, input_chord_durations, output_notes, output_note_durations, output_chords, output_chord_durations, n_vocab_notes, n_vocab_durations, n_vocab_chords, n_vocab_chord_durations = preprocess_sequences(all_files_notes, all_files_durations, all_files_chords, all_files_chord_durations, sequence_length, notes_mapping, chords_mapping, durations_mapping, chord_durations_mapping)
    train()