# generate_remi_music_transformer.py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tokenizer_remi import remi_to_midi, midi_to_remi, get_token_type

# Load model + vocab
from train_remi_music_transformer import RelativeMultiHeadAttention

custom_objects = {
    "RelativeMultiHeadAttention": RelativeMultiHeadAttention,
}

MODEL_PATH = "remi_music_transformer_large.h5"

print("Loading model...")
model = load_model(MODEL_PATH, custom_objects=custom_objects)

vocab = pickle.load(open("vocab.pkl", "rb"))
rev_vocab = pickle.load(open("rev_vocab.pkl", "rb"))
VOCAB = len(vocab)

print("Loaded vocab:", VOCAB)

# Token legality constraints
# token -> type-name ("Bar","Pos","Pitch","Dur","Vel")
def get_type_name(tok):
    if tok == "Bar":
        return "Bar"
    return tok.split("_")[0]


LEGAL_NEXT = {
    "Bar":  ["Pos"],
    "Pos":  ["Pitch"],
    "Pitch":["Dur"],
    "Dur":  ["Vel"],
    "Vel":  ["Pos", "Bar"]
}

def filter_illegal(pred, last_tok_str):
    last_type = get_type_name(last_tok_str)
    legal_types = LEGAL_NEXT[last_type]

    mask = np.zeros_like(pred)
    
    for idx, tok in rev_vocab.items():
        tname = get_type_name(tok)
        if tname in legal_types:
            mask[idx] = 1

    pred = pred * mask

    s = pred.sum()
    if s <= 1e-12:
        # fallback: uniform distribution ONLY over legal tokens
        legal_idx = np.where(mask == 1)[0]
        fallback = np.zeros_like(pred)
        fallback[legal_idx] = 1 / len(legal_idx)
        return fallback
    
    # normalize
    pred = pred / s
    pred = np.maximum(pred, 0)  # clamp negatives
    pred = pred / pred.sum()    # renormalize
    return pred

# GENERATE continuation
SEQ_LEN = 128

def pad_to_seq(tokens, target_len=SEQ_LEN-1):
    """Pad or trim tokens to exactly 127 length."""
    if len(tokens) >= target_len:
        return tokens[-target_len:]

    pad_len = target_len - len(tokens)
    return ["Bar"] * pad_len + tokens


def generate(seed_tokens, max_length=400, temperature=1.0):

    # ---- pad seed ----
    seed_tokens = pad_to_seq(seed_tokens)
    ids = [vocab[t] for t in seed_tokens]
    types = [get_token_type(t) for t in seed_tokens]

    for _ in range(max_length):
        x_tok = np.array(ids)[None, :]     # shape (1,127)
        x_type = np.array(types)[None, :]  # shape (1,127)

        pred = model.predict([x_tok, x_type], verbose=0)[0]

        # legality filter
        last_tok_str = seed_tokens[-1]
        pred = filter_illegal(pred, last_tok_str)

        # avoid negative / NaN
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.maximum(pred, 0)
        pred = pred / np.sum(pred)

        # temperature sampling
        logits = np.log(pred + 1e-9) / temperature
        exp = np.exp(logits)
        exp = np.maximum(exp, 0)
        exp = exp / exp.sum()

        nxt = np.random.choice(VOCAB, p=exp)
        tok_str = rev_vocab[nxt]

        seed_tokens.append(tok_str)
        ids.append(nxt)
        types.append(get_token_type(tok_str))

        # maintain window size
        # seed_tokens = seed_tokens[-(SEQ_LEN-1):]
        ids = ids[-(SEQ_LEN-1):]
        types = types[-(SEQ_LEN-1):]

    return seed_tokens

# OPTION 1 — Random generation (empty seed)
def generate_from_scratch():
    seed = ["Bar", "Pos_0"]  # valid minimal REMI start
    seq = generate(seed, max_length=300)
    remi_to_midi(seq, "generated_music.mid")
    print("Saved to generated_music.mid")


# OPTION 2 — Improvise based on your MIDI file
def improvise_from_midi(midi_path, keep=64, length=300):
    # --- 1. MIDI → REMI tokens ---
    seed = midi_to_remi(midi_path)

    if len(seed) < keep:
        keep = len(seed) - 1

    seed = seed[keep:keep*2]  # take first N tokens
    print("Using {} seed tokens".format(len(seed)))

    # --- 2. output only seed tokens as MIDI ---
    remi_to_midi(seed, MODEL_PATH[:-3] + "_input_seed.mid")
    print("Saved seed-only MIDI to input_seed.mid")

    # --- 3. generate continuation ---
    seq = generate(seed, max_length=length)
    print("Generated token length:",len(seq))
    seq = seq[-length:]
    # --- 4. save model continuation ---
    remi_to_midi(seq, MODEL_PATH[:-3] + "_jam_output.mid")
    print("Saved continuation to jam_output.mid")

    print("\n========================")
    print("Comparison files ready:")
    print(" - input_seed.mid")
    print(" - jam_output.mid")
    print("========================\n")

if __name__ == "__main__":
    # --- Option 1: random ---
    # generate_from_scratch()

    # --- Option 2: jamming with your MIDI ---
    improvise_from_midi(
        "./data/maestro-v3.0.0/2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi",
        keep=128,
        length=300
    )
