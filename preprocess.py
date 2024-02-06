import tensorflow as tf
from music21 import converter, note, chord
import glob
import numpy as np

def midi_to_notes(file_path):
    """_summary_

    Args:
        file_path (_str_): Return a compilation of notes from midi files in file_path
    """
    notes = []
    for file in glob.glob(file_path):

    try:
        midi = converter.parse(file)
        # Proceed with your code to flatten the score and extract notes...
    except Exception as e:
        print(f"Error parsing {file}: {e}")

    # Flatten the score to a single stream while keeping offsets
    notes_to_parse = midi.flat.notesAndRests

    for element in notes_to_parse:
        if isinstance(element, note.Note) or isinstance(element, chord.Chord):
            element_str = ' '.join(str(n) for n in element.pitches) if isinstance(element, chord.Chord) else str(element.pitch)
            notes.append(element_str)
    return notes

def preprocess(notes, sequence_length=50):
    n_vocab = len(set(notes)) # 326 unique notes and chords
    pitchnames = set(item for item in notes)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i+sequence_length]
        sequence_out = notes[i+sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_input = network_input / float(n_vocab)

    network_output = np.array(network_output)
    return n_vocab, network_input, network_output