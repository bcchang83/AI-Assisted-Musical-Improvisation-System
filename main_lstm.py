import glob
import numpy as np
import pretty_midi
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import random, pickle

# ===========================
# 1. Load and preprocess MIDI
# ===========================

def extract_melody_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                all_notes.append(note)
    
    # sort by start time
    all_notes.sort(key=lambda n: n.start)
    
    melody = []
    last_end = 0.0
    for note in all_notes:
        # if notes are not overlap
        if note.start >= last_end:
            melody.append([note.pitch, note.end - note.start])
            last_end = note.end
        elif note.pitch > melody[-1][0]:
            melody[-1] = [note.pitch, note.end - note.start]
            last_end = note.end

    return np.array(melody)

midi_files = glob.glob("./data/maestro-v3.0.0/**/*.midi", recursive=True)
print(f"Found {len(midi_files)} MIDI files")

seq_len = 150
step = 10
all_segments = []

for midi_path in midi_files[:20]:  # use a subset for faster training
    try:
        notes = extract_melody_notes(midi_path)
        if len(notes) > seq_len:
            for i in range(0, len(notes) - seq_len, step):
                segment = notes[i:i+seq_len]
                all_segments.append(segment)
    except Exception as e:
        print(f"Error reading {midi_path}: {e}")

all_segments = np.array(all_segments)
print("Total segments:", len(all_segments))

# ===========================
# 2. Normalize & Prepare Data
# ===========================

X = all_segments[:, :-1, :]
y = all_segments[:, -1, :]  # predict both pitch + duration

scaler = MinMaxScaler()
X_flat = X.reshape(-1, 2)
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
y_scaled = scaler.transform(y)

print("X:", X_scaled.shape, "y:", y_scaled.shape)

# ===========================
# 3. Build LSTM model
# ===========================

model = Sequential([
    LSTM(256, input_shape=(seq_len-1, 2), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(128, activation='relu'),
    Dense(2, activation='linear')  # predict pitch & duration (normalized)
])

model.compile(loss='mse', optimizer='adam')
model.summary()

# ===========================
# 4. Train model
# ===========================

model.fit(X_scaled, y_scaled, epochs=10, batch_size=64)
model.save("music_lstm_baseline_v2.h5")
pickle.dump(scaler, open("scaler_v2.pkl", "wb"))

# ===========================
# 5. Sampling utilities
# ===========================

def sample_with_temperature(pred, temperature=1.0):
    """Apply temperature sampling on pitch prediction."""
    pitch_prob = np.exp(pred / temperature) / np.sum(np.exp(pred / temperature))
    return np.random.choice(len(pitch_prob), p=pitch_prob)

# ===========================
# 6. Generate continuation
# ===========================

def generate_continuation(seed_seq, length=100, temperature=0.8):
    generated = []
    seq = seed_seq.copy()
    for _ in range(length):
        x_input = scaler.transform(seq.reshape(-1, 2)).reshape(1, seq_len-1, 2)
        pred = model.predict(x_input, verbose=0)[0]
        pred_inv = scaler.inverse_transform(pred.reshape(1, -1))[0]

        next_pitch = np.clip(int(pred_inv[0] + np.random.randn() * temperature), 21, 108)
        next_dur = max(0.05, pred_inv[1] + np.random.randn() * 0.02)
        next_note = [next_pitch, next_dur]
        generated.append(next_note)
        seq = np.vstack([seq[1:], next_note])
    return np.array(generated)

# ===========================
# 7. Export to MIDI
# ===========================

def notes_to_midi(notes, out_path="generated_v2.mid"):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    start = 0
    for pitch, dur in notes:
        note = pretty_midi.Note(
            velocity=80,
            pitch=int(np.clip(pitch, 21, 108)),
            start=start,
            end=start + dur
        )
        inst.notes.append(note)
        start += dur
    midi.instruments.append(inst)
    midi.write(out_path)
    print(f"✅ Saved generated music to {out_path}")

# ===========================
# 8. Generate & Save
# ===========================

seed = all_segments[random.randint(0, len(all_segments)-1), :-1, :]
generated_notes = generate_continuation(seed, length=150, temperature=0)
notes_to_midi(generated_notes, "maestro_continuation_v2.mid")
notes_to_midi(seed, "maestro_original_v2.mid")
