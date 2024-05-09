# Deep Melody LSTM

This repository contains a Python implementation of a hierarchical LSTM (Long Short-Term Memory) model for generating music sequences. The project utilizes the music21 library for processing MIDI files and extracting musical information, such as notes, chords, and key signatures.

## Data Preprocessing:
Processes MIDI files and extracts sequences of notes and chords for each measure.
Then converts the extracted sequences into integer representations suitable for the neural network input. Mapping function generates mappings between musical elements (notes, chords) and their integer representations.
## Model Architecture:
The hierarchical_lstm_model function defines the architecture of the hierarchical LSTM model, which consists of two parallel LSTM branches for note sequences and chord sequences. The outputs of these branches are concatenated and fed into another LSTM layer, allowing the model to learn the relationship between notes and chords.
## Training:
The train function sets up the model and preprocessed data, and initiates the training process using the TensorFlow/Keras API. The trained model is saved for later use in generating music sequences.
## Generation:
Takes the trained model, chords, and generates a specified number of new notes by sampling from the model's predictions. The generation process occurs on a measure-by-measure basis, where the model first predicts the chord for the next measure, and then generates the sequence of notes based on the predicted chord and the previous sequence of notes and chords.

## To do 
- [X ] Train on a larger dataset
- [X] Preprocess the dataset into beats instead of measures to have more granular training over the notes
- [ ] Explore different neural network architectures specifically multi-head attention mechanisms
- [ ] Incorporate additional musical features such as dynamics, meter and instrument combinations
- [ ] Deploy the model to a web application where users can input seed notes 
