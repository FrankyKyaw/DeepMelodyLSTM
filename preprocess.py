import music21
from music21 import converter, note, instrument, harmony, midi, roman, key, stream
from music21 import environment, configure
import music21.chord as chord_module
import music21.note as note_module
import glob
import json
import numpy as np

def extract_notes_with_music21(file_path):
    # Load the MIDI file
    midi = converter.parse(file_path)

    original_key = midi.analyze('key')
    if str(original_key) != "C major":      
        transposed_key = key.Key('C')
        interval = music21.interval.Interval(original_key.tonic, transposed_key.tonic)
        midi = midi.transpose(interval)
    else:
        transposed_key = original_key
    all_chords = []
    all_notes = []
    # for element in midi.recurse():
    #     if any(cls in ['Unpitched', 'PercussionChord'] for cls in element.classes):
    #         rest = note_module.Rest()
    #         rest.duration = element.durat|ion
    #         if element.activeSite:
    #             element.activeSite.replace(element, rest)
    chords = midi.chordify()
    for measure in chords.getElementsByClass('Measure'):
        measure_notes = {}
        for note in measure.notesAndRests:
            beat = note.beat
            if beat not in measure_notes:
                measure_notes[beat] = []
            measure_notes[beat].append(note)
        note_durations = {}
        for beat, notes in measure_notes.items():
            beat_notes = []
            for note in notes:
                if isinstance(note, chord_module.Chord):
                    for single_note in note.pitches:
                        midi_value = single_note.midi
                        note_durations[midi_value] = note.duration.quarterLength
                        beat_notes.append(midi_value)
                elif isinstance(note, note_module.Note):
                    midi_value = note.pitch.midi
                    note_durations[midi_value] = note.duration.quarterLength
                    beat_notes.append(midi_value)
                elif isinstance(note, note_module.Rest):
                    beat_notes.append('R')
            all_notes.append(beat_notes)
        if note_durations:
            sorted_notes = sorted(note_durations, key=note_durations.get, reverse=True)
            if len(sorted_notes) >= 3:
                freq = chord_module.Chord(sorted_notes[:4])
                rn = roman.romanNumeralFromChord(freq, transposed_key)
                simplified_name = simplify_roman_name(rn)
            else:
                simplified_name = 'rest_or_no_chord'

        # Append the chords 4 times for each beat
        all_chords.append([simplified_name] * 4)
    return all_chords, all_notes

def simplify_roman_name(roman_numeral):
    """
    Simplify roman numeral chord names.
    """
    simplified_name = roman_numeral.romanNumeral.upper()
    return simplified_name
    

def preprocess_sequences(notes, chords, sequence_length, notes_mapping, chords_mapping):
#     with open(single_file_dataset, "r") as fp:
#         all_notes = fp.read()            
    encoded_notes = convert_to_int(notes, notes_mapping)
    encoded_chords = convert_to_int(chords, chords_mapping)
    print(len(encoded_notes), len(encoded_chords))
    network_input_notes = []
    network_input_chords = []
    network_output_notes = []
    network_output_chords = []
    # Duplicate each chord 4 times
    encoded_chords_duplicated = [chord for chord in encoded_chords for _ in range(4)]
    for i in range(len(encoded_chords_duplicated) - sequence_length):
        sequence_in_notes = encoded_notes[i:i + sequence_length]
        sequence_out_note = encoded_notes[i + sequence_length]
        sequence_in_chords = encoded_chords_duplicated[i:i + sequence_length]
        sequence_out_chord = encoded_chords_duplicated[i + sequence_length]
        
        network_input_notes.append(sequence_in_notes)
        network_input_chords.append(sequence_in_chords)
        network_output_notes.append(sequence_out_note)
        network_output_chords.append(sequence_out_chord)
    print(encoded_notes[:100])
    n_vocab_notes = len(set(encoded_notes))
    n_vocab_chords = len(set(encoded_chords))
    return np.array(network_input_notes), np.array(network_input_chords), np.array(network_output_notes), np.array(network_output_chords), n_vocab_notes, n_vocab_chords
    
def create_mappings(items, file_path):
    unique_items = []
    for item in items:
        unique_items.append('_'.join(str(i) for i in item))
    mappings = {item: number for number, item in enumerate(set(unique_items))}
    with open(file_path, "w") as file:
        json.dump(mappings, file)
    return mappings

def convert_to_int(items, mapping_file):
    int_notes = []
    with open(mapping_file, "r") as fp:
        mappings = json.load(fp)
    for item in items:
        if isinstance(item, list):
            item_str = '_'.join(str(i) for i in item)
        else:
            item_str = item
        
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

    a = create_mappings(all_files_notes, note_mapping_path)
    b = create_mappings(all_files_chords, chord_mapping_path)
    sequence_length = 64

    # Prepare input and output sequences
    network_input_notes, network_input_chords, network_output_notes, network_output_chords, n_vocab_notes, n_vocab_chords = preprocess_sequences(all_files_notes, all_files_chords, sequence_length, note_mapping_path, chord_mapping_path)

if __name__ == "__main__":
    main()