import tensorflow as tf
from music21 import stream
import music21.chord as chord_module
import music21.note as note_module
import numpy as np
import json
from keras.models import load_model
import ast



def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def temperature_sampling(predictions, temperature=1.0):
    scaled = np.log(predictions + 1e-7) / temperature  
    probabilities = softmax(scaled)
    choice = np.random.choice(len(probabilities), p=probabilities)
    return choice
    
def predict_notes(model_path, starting_notes, starting_melody_durations, starting_chords, starting_chord_durations, sequence_length, n_notes=300, temperature=1.0):
    model = load_model(model_path)

    prediction_output_notes = []
    prediction_output_melody_durations = []
    prediction_output_chords = []
    prediction_output_chord_durations = []
    
    with open(note_mapping_path, "r") as f:
        note_mappings = json.load(f)
    with open(note_duration_mapping_path, "r") as f:
        melody_duration_mappings = json.load(f)
    with open(chord_duration_mapping_path, "r") as f:
        chord_duration_mappings = json.load(f)
    with open(chord_mapping_path, "r") as f:
        chord_mappings = json.load(f)

    reverse_note_mappings = {value: key for key, value in note_mappings.items()}
    reverse_melody_duration_mappings = {value: key for key, value in melody_duration_mappings.items()}
    reverse_chord_duration_mappings = {value: key for key, value in chord_duration_mappings.items()}
    reverse_chord_mappings = {value: key for key, value in chord_mappings.items()}

    starting_notes_int = convert_to_int(starting_notes, note_mappings)
    starting_melody_durations_int = convert_to_int(starting_melody_durations, melody_duration_mappings)
    starting_chords_int = convert_to_int(starting_chords, chord_mappings)
    starting_chord_durations_int = convert_to_int(starting_chord_durations, chord_duration_mappings)


    for _ in range(n_notes):
        input_sequence_notes = np.array([starting_notes_int[-sequence_length:]])
        input_sequence_melody_durations = np.array([starting_melody_durations_int[-sequence_length:]])
        input_sequence_chords = np.array([starting_chords_int[-sequence_length:]])
        input_sequence_chord_durations = np.array([starting_chord_durations_int[-sequence_length:]])

        prediction = model.predict([input_sequence_notes, input_sequence_melody_durations, input_sequence_chords, input_sequence_chord_durations], verbose=0)
        next_note = temperature_sampling(prediction[0][0], temperature)
        next_melody_duration = temperature_sampling(prediction[1][0], temperature)
        next_chord = temperature_sampling(prediction[2][0], temperature)
        next_chord_duration = temperature_sampling(prediction[3][0], temperature)

        # Update the sequences with the predicted values
        starting_notes_int.append(next_note)
        starting_melody_durations_int.append(next_melody_duration)
        starting_chords_int.append(next_chord)
        starting_chord_durations_int.append(next_chord_duration)

        # Convert predictions to midi notes and durations
        prediction_output_notes.append(reverse_note_mappings.get(next_note, "Unknown"))
        prediction_output_melody_durations.append(reverse_melody_duration_mappings.get(next_melody_duration, "Unknown"))
        prediction_output_chords.append(reverse_chord_mappings.get(next_chord, "Unknown"))
        prediction_output_chord_durations.append(reverse_chord_duration_mappings.get(next_chord_duration, "Unknown"))

    return prediction_output_notes, prediction_output_melody_durations, prediction_output_chords, prediction_output_chord_durations

def create_midi(predictions_notes, predictions_durations, predictions_chords, predictions_chord_durations, file_name="output.mid"):
    s = stream.Score()
    melody_part = stream.Part()
    chord_part = stream.Part()
    
    for i, (notes, dur, chords, chord_dur) in enumerate(zip(predictions_notes, predictions_durations, predictions_chords, predictions_chord_durations)):
        note_list = notes.split('_')
        duration_list = dur.split('_')
        chord_list = chords.split('_')
        chord_duration_list = chord_dur.split('_')

        if len(duration_list) < len(note_list):
            last_duration = duration_list[-1] if duration_list else 1
            duration_list.extend([last_duration] * (len(note_list) - len(duration_list)))

        if len(chord_duration_list) < len(chord_list):
            last_chord_duration = chord_duration_list[-1] if chord_duration_list else 1
            chord_duration_list.extend([last_chord_duration] * (len(chord_list) - len(chord_duration_list)))
            
        for j in range(len(note_list)):
            if note_list[j] == '<end>':
                break
            elif note_list[j] == 'Rest':
                n = note_module.Rest()
            else:
                 try:
                    note_value = ast.literal_eval(note_list[j])
                    if isinstance(note_value, list):
                        pitches = [int(p) for p in note_value]
                        n = chord_module.Chord(pitches)
                    else:
                        n = note_module.Note()
                        n.pitch.midi = int(note_value)
                 except: 
                    n = note_module.Note()
                    n.pitch.midi = int(note_list[j])
            n.duration.quarterLength = float(reverse_duration_mapping.get(int(duration_list[j]), 1.0))
        
            melody_part.append(n)
            
        for k in range(len(chord_list)):
            chord_data = chord_list[k]
            if chord_list[k] == '<end>':
                break
            elif chord_list[k] == 'Rest':
                c = note_module.Rest()
            else:
                chord_pitches = chord_list[k].split('_')
                for chord_str in chord_pitches:
                    if chord_str == 'Rest':
                        c = note_module.Rest()
                    else:
                        chord_pitches = chord_str.strip('[]').split(',')
                        chord_pitches = [int(p) for p in chord_pitches]
                        c = chord_module.Chord(chord_pitches)
            print(float(reverse_duration_mapping.get(int(chord_duration_list[k]))))
            c.duration.quarterLength = float(reverse_duration_mapping.get(int(chord_duration_list[k]), 1.0))
            chord_part.append(c)
    s.insert(0, melody_part)
    s.insert(0, chord_part)
    s.write('midi', fp=file_name)   
    


if __name__ == "__main__":
    output_path = "output.mid"
    seed = 100
    starting_notes = all_files_notes[seed:seed+sequence_length]
    starting_melody_durations = all_files_durations[seed:seed+sequence_length]
    starting_chords = all_files_chords[seed:seed+sequence_length]
    starting_chord_durations = all_files_chord_durations[seed:seed+sequence_length]
    
    # Predict the notes
    prediction_notes, prediction_melody_durations, prediction_chords, prediction_chord_durations = predict_notes(MODEL_PATH, starting_notes, starting_melody_durations, starting_chords, starting_chord_durations, sequence_length, n_notes=200, temperature=1.0)

    # Create the MIDI file
    create_midi(prediction_notes, prediction_melody_durations, prediction_chords, prediction_chord_durations, output_path)
    print("MIDI file created")