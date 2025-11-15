# transformer_chord_event_model.py

import glob
import numpy as np
import pretty_midi
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import random, pickle

# ===========================
# 1. Extract chord events (same as before)
# ===========================
def midi_to_chord_events(midi_path, chord_merge_ms=0.03):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    if not notes:
        return None

    # Sort notes by start time
    notes.sort(key=lambda n: n.start)
    events = []
    group = [notes[0]]
    for note in notes[1:]:
        if note.start - group[-1].start < chord_merge_ms:
            group.append(note)
        else:
            pitches = [n.pitch for n in group]
            avg_dur = np.mean([n.end - n.start for n in group])
            events.append([pitches, group[0].start, avg_dur])
            group = [note]
    if group:
        pitches = [n.pitch for n in group]
        avg_dur = np.mean([n.end - n.start for n in group])
        events.append([pitches, group[0].start, avg_dur])

    # Convert to delta-time representation
    result = []
    prev_time = 0.0
    for pitches, start, dur in events:
        delta = start - prev_time
        result.append([pitches, delta, dur])
        prev_time = start
    return result


# ===========================

# 2. Convert chord events to numeric training data

# ===========================

def events_to_vectors(events):
    """Convert each chord event to 130D vector:
       [128 pitch one-hot + delta_time + duration]"""
    vecs = []
    for pitches, delta, dur in events:
        v = np.zeros(130)
        for p in pitches:
            if 0 <= p < 128:
                v[p] = 1.0
        v[128] = delta
        v[129] = dur
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)

# ===========================

# 3. Dataset

# ===========================

midi_files = glob.glob("./data/maestro-v3.0.0/**/*.midi", recursive=True)
print(f"Found {len(midi_files)} files")

seq_len = 64
step = 8
segments = []

for path in midi_files[:50]:
    try:
        events = midi_to_chord_events(path)
        if events is None or len(events) <= seq_len:
            continue
        vecs = events_to_vectors(events)
        for i in range(0, len(vecs) - seq_len, step):
            segments.append(vecs[i:i+seq_len])
    except Exception as e:
        print(f"Error reading {path}: {e}")

segments = np.array(segments)
print("✅ Segments:", segments.shape)

X = segments[:, :-1, :]
y = segments[:, -1, :]

scaler = MinMaxScaler()
time_feats = X[:, :, 128:].reshape(-1, 2)
scaler.fit(time_feats)
X[:, :, 128:] = scaler.transform(time_feats).reshape(X[:, :, 128:].shape)
y[:, 128:] = scaler.transform(y[:, 128:].reshape(-1, 2))
pickle.dump(scaler, open("scaler_chord_event.pkl", "wb"))

# ===========================

# 4. Transformer Model

# ===========================

def transformer_block(x, num_heads=8, ff_dim=512, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(x.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

input_dim = 130
inputs = Input(shape=(seq_len-1, input_dim))
x = transformer_block(inputs)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(input_dim, activation='sigmoid')(x)
model = Model(inputs, outputs)
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4))
model.summary()

# ===========================

# 5. Train

# ===========================

model.fit(X, y, epochs=20, batch_size=32)
model.save("transformer_chord_event_model.h5")

# ===========================

# 6. Generate continuation

# ===========================

def generate_chord_events(seed, length=64, temperature=0.8):
    seq = seed.copy()
    generated = []
    for _ in range(length):
        x = seq.reshape(1, seq.shape[0], seq.shape[1])
        pred = model.predict(x, verbose=0)[0]
        pitch_probs = pred[:128]
        delta_norm, dur_norm = pred[128], pred[129]
        delta, dur = scaler.inverse_transform([[delta_norm, dur_norm]])[0]
        probs = np.log(pitch_probs + 1e-8) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        chord_vec = (np.random.rand(128) < probs).astype(np.float32)
        chord_pitches = np.where(chord_vec > 0.5)[0].tolist()
        generated.append([chord_pitches, delta, dur])
        new_vec = np.zeros(130)
        new_vec[:128] = chord_vec
        new_vec[128], new_vec[129] = pred[128], pred[129]
        seq = np.vstack([seq[1:], new_vec])
    return generated

# ===========================

# 7. Convert to MIDI

# ===========================

def chord_events_to_midi(events, out_path="generated_transformer.mid"):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for pitches, delta, dur in events:
        t += delta
        for p in pitches:
            inst.notes.append(pretty_midi.Note( velocity=90,
                                                pitch=int(p),
                                                start=t,
                                                end=t + dur
                                                ))
            midi.instruments.append(inst)
            midi.write(out_path)
    print(f"🎶 Saved MIDI: {out_path}")

# ===========================

# 8. Generate sample

# ===========================

seed = X[0]
generated_events = generate_chord_events(seed, length=128)
chord_events_to_midi(generated_events, "transformer_generated.mid")
def vectors_to_events(vectors, scaler):
    events = []
    for v in vectors:
        pitches = np.where(v[:128] > 0.5)[0].tolist()
        delta_norm, dur_norm = v[128], v[129]
        delta, dur = scaler.inverse_transform([[delta_norm, dur_norm]])[0]
        events.append([pitches, delta, dur])
    return events

# seed to event and reconstruct
seed_events = vectors_to_events(seed, scaler)
chord_events_to_midi(seed_events, "transformer_original.mid")
print("✅ Done! Transformer chord event model trained successfully.")
