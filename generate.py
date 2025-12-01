import torch
import numpy as np
import pretty_midi
from train_lstm import MelodyLSTM

# ---- Load model ----
checkpoint = torch.load("lstm_model.pt", map_location="cpu")
idx2pitch = checkpoint["idx2pitch"]
pitch2idx = checkpoint["pitch2idx"]
vocab_size = len(idx2pitch)

model = MelodyLSTM(vocab_size, 256)
model.load_state_dict(checkpoint["model"])
model.eval()

# ---- Generate ----
seed = [60, 62, 64, 65, 67, 69, 71, 72]  # C-major ascending
seq = [pitch2idx[p] for p in seed if p in pitch2idx]

generated = seq.copy()
length = 200  # number of predicted notes

for _ in range(length):
    inp = torch.tensor([generated[-50:]], dtype=torch.long)
    with torch.no_grad():
        out = model(inp)
        next_pitch = torch.argmax(out[0, -1]).item()
    generated.append(next_pitch)

notes = [idx2pitch[i] for i in generated]

# ---- Export MIDI ----
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)
start = 0
for p in notes:
    note = pretty_midi.Note(velocity=80, pitch=int(p), start=start, end=start+0.5)
    instrument.notes.append(note)
    start += 0.5
midi.instruments.append(instrument)
midi.write("ai_jam.mid")
print("🎶 Generated AI jam saved as ai_jam.mid")
