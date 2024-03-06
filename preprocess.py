import music21
from music21 import converter, note, instrument, harmony, midi, roman, key, stream
from music21 import environment, configure
import music21.chord as chord_module
import music21.note as note_module
import glob
import json

def extract_notes_with_music21(file_path):
    # Load the MIDI file
    midi = converter.parse(file_path)

    original_key = midi.analyze('key')
    transposed_key = key.Key('C')
    interval = music21.interval.Interval(original_key.tonic, transposed_key.tonic)
    transposed_stream = midi.transpose(interval)

    # Dictionary to store notes by their offset
    all_chords = []
    all_notes = []
    for element in midi.recurse():
        if any(cls in ['Unpitched', 'PercussionChord'] for cls in element.classes):
            rest = note_module.Rest()
            rest.duration = element.duration
            if element.activeSite:
                element.activeSite.replace(element, rest)

    chords = transposed_stream.chordify()
    for measure in chords.getElementsByClass('Measure'):
        measure_lst = []
        note_durations = {}
        for element in measure.notes:
            if isinstance(element, chord_module.Chord):
                symbol = '.'.join(str(n.midi) for n in element.pitches)
                for note in element.pitches:
                    note_name = note.nameWithOctave
                    note_durations[note_name] = note_durations.get(note_name, 0) + element.duration.quarterLength
            elif isinstance(element, note.Rest):
                symbol = "r"
            else: 
                continue
            measure_lst.append(symbol)

        if note_durations:
            sorted_notes = sorted(note_durations, key=note_durations.get, reverse=True)[:4]
            freq = chord_module.Chord(sorted_notes)
            rn = roman.romanNumeralFromChord(freq, transposed_key)
            simplified_name = simplify_roman_name(rn)
        
        else:
            simplified_name = 'rest_or_no_chord'

        # Append measure list to the correct key in the dictionary
        all_chords.append(simplified_name)
        all_notes.append(measure_lst)
    return all_chords, all_notes

def simplify_roman_name(roman_numeral):
    """
    Simplify roman numeral chord names.
    """
    simplified_name = roman_numeral.figure
    if roman_numeral.inversion() == 0:
        return simplified_name
    else:
        # Add inversion to the chord name
        return f"{simplified_name}/{roman_numeral.bass().name}"
    

def preprocess_sequences(items, sequence_length, mappings_file):
#     with open(single_file_dataset, "r") as fp:
#         all_notes = fp.read()        
    encoded_notes = convert_to_int(items, mappings_file)
    network_input = []
    network_output = []
    for i in range(len(items) - sequence_length):
        sequence_in = items[i:i + sequence_length]
        sequence_out = items[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)
    n_vocab = len(set(items))
    return np.array(network_input), np.array(network_output), n_vocab
    
def create_mappings(items, file_path):
    unique_items = {'_'.join(item) for item in items}
    mappings = {item: number for number, item in enumerate(unique_items)}
    with open(file_path, "w") as file:
        json.dump(mappings, file)
    return mappings

def convert_to_int(items, mapping_file):
    int_notes = []
    with open(mapping_file, "r") as fp:
        mappings = json.load(fp)
    for item in items:
        item_str = '_'.join(item) if isinstance(item, list) or isinstance(item, str) else item
        int_notes.append(mappings.get(item_str, -1))
    return int_notes


def main():
    file_path = "mozart/*.mid"
    midi_files = glob.glob(file_path)
    all_files_chords = []
    all_files_notes = []
    for file in midi_files:
        all_chords, all_notes = extract_notes_with_music21(file)
        all_files_chords.extend(all_chords)
        all_files_notes.extend(all_notes)

    note_mapping_path = "note_mappings.txt"
    chord_mapping_path = "chord_mappings.txt"
    # Convert notes and chords to integer sequences
    notes_int = convert_to_int(all_files_notes, note_mapping_path)
    chords_int = convert_to_int(all_files_chords, chord_mapping_path)

    # Prepare input and output sequences
    network_input_notes, network_output_notes, n_vocab_notes = preprocess_sequences(notes_int, sequence_length, note_mapping_path)
    network_input_chords, network_output_chords, n_vocab_chords = preprocess_sequences(chords_int, sequence_length, chord_mapping_path)

if __name__ == "__main__":
    main()