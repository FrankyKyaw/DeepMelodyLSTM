import tensorflow as tf
from music21 import converter, stream, roman, key
import music21
import music21.chord as chord_module
import music21.note as note_module
import glob
import numpy as np
import json
from pathlib import Path

duration_mapping = {
    0.25: 0,  # Sixteenth note
    0.5: 1,   # Eighth note
    0.75: 2,
    1.0: 3,   # Quarter note
    1.25: 4,
    1.5: 5,   # Dotted quarter note
    1.75: 6,
    2.0: 7,   # Half note
    2.25: 8,
    2.5: 9,   
    2.75: 10,
    3.0: 11,   
    3.25: 12,  
    3.5: 13,   
    3.75: 14, 
    4.0: 15    # Whole note
}

def get_midi_files_pathlib(directory):
    path = Path(directory)
    files_lower = list(path.glob('*.mid'))
    files_upper = list(path.glob('*.MID'))

    all_files = files_lower + files_upper
    return all_files



def get_duration(duration, duration_mapping=duration_mapping):
    min_diff = float('inf')
    nearest_duration = None
    for key in duration_mapping:
        diff = abs(duration - key)
        if diff < min_diff:
            min_diff = diff
            nearest_duration = key
    return duration_mapping[nearest_duration]
    
def extract_midi_file(midi_file):
    midi = converter.parse(midi_file)
    original_key = midi.analyze('key')
    if str(original_key) != "C major":      
        transposed_key = key.Key('C')
        interval = music21.interval.Interval(original_key.tonic, transposed_key.tonic)
        midi = midi.transpose(interval)
    else:
        transposed_key = original_key
            
    melody_stream = stream.Score()
    chord_stream = stream.Score()
    
    melody_part = midi.parts[0]
    melody_stream.append(melody_part)
    
    if not melody_stream.hasMeasures():
        melody_stream.makeMeasures(inPlace=True)
        
    for part in midi.parts[1:]:
        chord_stream.append(part)
    chords = chord_stream.chordify()

    melody_measures = list(melody_part.measures(0, None))
    chord_measures = list(chords.measures(0, None))
    
    melody_notes = []
    melody_duration = []
    chord_notes = []
    chord_duration = []
    

    for melody_measure, chord_measure in zip(melody_measures, chord_measures):
        melody, mel_duration = extract_measure_info(melody_measure)
        chord, ch_duration  = extract_measure_info(chord_measure)
        
        melody_notes.append(melody)
        melody_duration.append(mel_duration)
        chord_notes.append(chord)
        chord_duration.append(ch_duration)
        
    melody_notes.append(['<end>'])
    melody_duration.append([0])  
    chord_notes.append(['<end>'])
    chord_duration.append([0])
    
    return melody_notes, melody_duration, chord_notes, chord_duration
    
def extract_measure_info(measure):
    notes = []
    durations = []
    for element in measure.notesAndRests:
        if isinstance(element, note_module.Note):
            notes.append(element.pitch.midi)
            mapped_duration = get_duration(float(element.duration.quarterLength), duration_mapping)
            durations.append(mapped_duration)
        elif isinstance(element, chord_module.Chord):
            chord_notes = [p.midi for p in element.pitches]
            notes.append(chord_notes)
            mapped_duration = get_duration(float(element.duration.quarterLength), duration_mapping)
            durations.append(mapped_duration)
        elif isinstance(element, note_module.Rest):
            notes.append('Rest')
            mapped_duration = get_duration(float(element.duration.quarterLength), duration_mapping)
            durations.append(mapped_duration)
    return notes, durations

def create_mappings(items_list, mapping_path):
    unique_items = set() 
    for items in items_list:
        item_str = '_'.join(str(i) for i in items)
        unique_items.add(item_str)  
    
    unique_items = list(unique_items)
    
    mappings = {item: number for number, item in enumerate(unique_items)}
    with open(mapping_path, "w") as file:
        json.dump(mappings, file)
    return mappings

def convert_to_int(items, mapping):
    int_notes = []
    
    for item in items:
        if isinstance(item, list):
            item_str = '_'.join(str(i) for i in item)
        else:
            item_str = item
        
        int_notes.append(mapping.get(item_str, -1))
    return int_notes

def main():
    reverse_duration_mapping = {value: key for key, value in duration_mapping.items()}
    directory = 'dataset/choral-dataset/'
    midi_files = get_midi_files_pathlib(directory)

    all_files_notes = []
    all_files_durations = []
    all_files_chords = []
    all_files_chord_durations = []
    count = 0
    for file in midi_files:
        notes, durations, chords, chord_durations = extract_midi_file(file)
        if notes is not None and durations is not None and chords is not None and chord_durations is not None:
            all_files_notes.extend(notes)
            all_files_durations.extend(durations)
            all_files_chords.extend(chords)
            all_files_chord_durations.extend(chord_durations)
            count += 1
            print(count)
        else:
            print(f"Skipping {file} due to errors in extraction.")

    note_mapping_path = "mappings_model/note_mappings.json"
    note_duration_mapping_path = "mappings_model/note_duration_mappings.json"
    chord_mapping_path = "mappings_model/chord_mappings.json"
    chord_duration_mapping_path = "mappings_model/chord_duration_mappings.json"

    notes_mapping = create_mappings(all_files_notes, note_mapping_path)
    durations_mapping = create_mappings(all_files_durations, note_duration_mapping_path)
    chords_mapping = create_mappings(all_files_chords, chord_mapping_path)
    chord_durations_mapping = create_mappings(all_files_chord_durations, chord_duration_mapping_path)


if __name__ == "__main__":
    main()