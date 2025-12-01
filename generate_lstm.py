import pretty_midi
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# === Load model and scaler ===
model = load_model("music_lstm_baseline.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# === Helper functions ===
def extract_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitch = note.pitch
                dur = note.end - note.start
                notes.append([pitch, dur])
    return np.array(notes)

def notes_to_midi(notes_human, notes_ai, out_path="jam_session.mid"):
    midi = pretty_midi.PrettyMIDI()
    
    # Track 1: Human input
    inst_h = pretty_midi.Instrument(program=0, name="Human")
    t = 0
    for p, d in notes_human:
        inst_h.notes.append(pretty_midi.Note(velocity=90, pitch=int(p), start=t, end=t+d))
        t += d
    midi.instruments.append(inst_h)

    # Track 2: AI response
    inst_ai = pretty_midi.Instrument(program=0, name="AI")
    t = t + 0.2  # slight delay
    for p, d in notes_ai:
        inst_ai.notes.append(pretty_midi.Note(velocity=100, pitch=int(p), start=t, end=t+d))
        t += d
    midi.instruments.append(inst_ai)
    
    midi.write(out_path)
    print(f"✅ Saved jam session to {out_path}")

def generate_continuation(seed_seq, model, scaler, length=80, temperature=0.8):
    generated = []
    seq = seed_seq.copy()
    for _ in range(length):
        x_input = scaler.transform(seq.reshape(-1, 2)).reshape(1, seq.shape[0], 2)
        pred = model.predict(x_input, verbose=0)
        probs = np.log(pred[0] + 1e-8) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        next_pitch = np.random.choice(range(128), p=probs)
        next_dur = np.random.uniform(0.1, 0.4)
        generated.append([next_pitch, next_dur])
        seq = np.vstack([seq[1:], [next_pitch, next_dur]])
    return np.array(generated)

# === Example usage ===
human_input = extract_notes("./data/maestro-v3.0.0/2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi")  # your melody snippet
seed = human_input[-128:]  # take last 50 notes as seed

ai_response = generate_continuation(seed, model, scaler, length=300, temperature=1.0)
notes_to_midi(human_input, ai_response, "lstm_jam_output.mid")
