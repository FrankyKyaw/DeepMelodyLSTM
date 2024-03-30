from tensorflow.keras.models import load_model
from train_model import generate_model
import numpy as np
import music21.chord as chord_module
import music21.note as note_module
from music21 import stream
import json

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def temperature_sampling(predictions, temperature=1.0):
    scaled = np.log(predictions) / temperature
    probabilities = softmax(scaled)
    choice = np.random.choice(len(probabilities), p=probabilities) # choose according to probability
    return choice

def predict_notes(model_path, starting_notes, starting_chords, sequence_length, n_notes=300, temperature=1.0):
    # Load the model
    model = load_model(model_path)

    prediction_output_notes = []  
    
    starting_chords_int = convert_to_int(starting_chords, chord_mapping_path)
    starting_notes_int = convert_to_int(starting_notes, note_mapping_path)
    with open(note_mapping_path, "r") as fp:
        note_mappings = json.load(fp)
        
    reverse_note_mappings = {value: key for key, value in note_mappings.items()}
    
    for _ in range(n_notes):
        input_sequence_chords = np.array(starting_chords_int[-sequence_length:]).reshape(1, sequence_length)
        input_sequence_notes = np.array(starting_notes_int[-sequence_length:]).reshape(1, sequence_length)

        prediction_notes, prediction_chords = model.predict([input_sequence_chords, input_sequence_notes], verbose=0)
        next_note = temperature_sampling(prediction_notes[0], temperature)  
        next_chord = temperature_sampling(prediction_chords[0], temperature) 
        
        starting_notes_int.append(next_note)
        starting_chords_int.append(next_chord)
        
        prediction_output_notes.append(reverse_note_mappings[next_note])


    return prediction_output_notes

def create_midi(predicted_notes, output_path=output_path):
    s = stream.Stream()
    for segment in predicted_notes:
        elements = segment.split('_')
        if len(elements) == 1 and elements[0].isdigit(): 
            n = note_module.Note()
            n.pitch.midi = int(elements[0])
            n.quarterLength = 1  
            s.append(n)
        else: 
            chord_pitches = [int(p) for p in elements if p.isdigit()]
            if chord_pitches:  
                c = chord_module.Chord(chord_pitches)
                c.quarterLength = 1
                s.append(c)
    s.write('midi', fp=output_path)


if __name__ == "__main__":
    starting_chords = all_files_chords[200:216]
    starting_notes = all_files_notes[200:216]
    # starting_notes = ["53.77_57.77_60_65_64_63", "54.61.70_53.61.70_51_49_48.58.61_46.58.61", "54.61.70_53.61.70_51_49_48.58.61_46.58.61", "51.55.72_51.60.79_51.55.75_51.60.74_51.55.72_51.60.70"]
    sequence_length = 16
    n_notes = 60
    predicted_notes = predict_notes(MODEL_PATH, starting_notes, starting_chords, sequence_length, n_notes)
    create_midi(predicted_notes, output_path)


