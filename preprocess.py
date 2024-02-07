from music21 import converter, note, chord
import glob
import numpy as np
import json

def midi_to_notes(file_path, time_step=0.25, destination_path='output.txt'):
    """_summary_

    Args:
        file_path (_str_): Return a compilation of notes from midi files in file_path
    """
    all_notes = []
    for file in glob.glob(file_path):
        notes = []
        try:
            midi = converter.parse(file)
            # Proceed with your code to flatten the score and extract notes...
        except Exception as e:
            print(f"Error parsing {file}: {e}")

        # Flatten the score to a single stream while keeping offsets
        notes_to_parse = midi.flat.notesAndRests

        for element in notes_to_parse:
            steps = int(element.duration.quarterLength / time_step)
            if isinstance(element, note.Note):
                symbol = element.pitch.midi
            elif isinstance(element, chord.Chord):
                symbol = '.'.join(str(n.midi) for n in element.pitches)
            elif isinstance(element, note.Rest):
                symbol = "r"
            else: 
                continue
            for step in range(steps):
                if step == 0:
                    notes.append(str(symbol))
                else:
                    notes.append("_")

        all_notes.extend(notes + ["<end>"])
        # all_notes.append(" ".join(notes) + " " + "<end>")

    with open(destination_path, "w") as file:
        file.write(" ".join(all_notes))
    return all_notes

def preprocess_notes(single_file_dataset="output.txt", mappings_file="mappings.json", sequence_length=50):
    # n_vocab = len(set(notes)) # 326 unique notes and chords
    # pitchnames = set(item for item in notes)

    # note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    with open(single_file_dataset, "r") as fp:
        all_notes = fp.read()
    encoded_notes = convert_to_int(all_notes, mappings_file)

    network_input = []
    network_output = []
    n_vocab = len(set(encoded_notes))

    for i in range(0, len(encoded_notes) - sequence_length, 1):
        network_input.append(encoded_notes[i:i+sequence_length])
        network_output.append(encoded_notes[i+sequence_length])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_input = network_input / float(n_vocab)

    network_output = np.array(network_output)
    return n_vocab, network_input, network_output

def create_mappings(notes, file_path):
    mappings = {}
    
    pitchnames = set(item for item in notes)
    for number, note in enumerate(pitchnames):
        mappings[note] = number
    with open(file_path, "w") as file:
        json.dump(mappings, file)

def convert_to_int(notes, mapping_file):
    int_notes = []

    with open(mapping_file, "r") as fp:
        mappings = json.load(fp)

    for note in notes:
        int_notes.append(mappings[note])
    return int_notes

def main():
    file_path = "../datasets/midi_songs/*.mid"
    output_path = "output.txt"
    notes = midi_to_notes(file_path, 0.25, output_path)
    create_mappings(notes, "mappings.json")

if __name__ == "__main__":
    main()