# generate_remi.py
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tokenizer_remi import midi_to_remi, remi_to_midi, get_token_type

# -----------------------
# Load model & vocab
# -----------------------
model = load_model("remi_transformer.h5")
vocab = pickle.load(open("vocab.pkl", "rb"))
rev_vocab = pickle.load(open("rev_vocab.pkl", "rb"))
VOCAB = len(vocab)

TOKEN_TYPES = {
    "Bar": 0,
    "Pos": 1,
    "Pitch": 2,
    "Dur": 3,
    "Vel": 4
}

LEGAL_NEXT = {
    "Bar": ["Pos"],
    "Pos": ["Pitch"],
    "Pitch": ["Dur"],
    "Dur": ["Vel"],
    "Vel": ["Pos", "Bar"]
}

def get_name(tok):
    return "Bar" if tok == "Bar" else tok.split("_")[0]

def filter_illegal(pred, last_type_name):
    mask = np.zeros(VOCAB, dtype=np.float32)
    for idx in range(VOCAB):
        tok = rev_vocab[idx]
        name = get_name(tok)
        if name in LEGAL_NEXT[last_type_name]:
            mask[idx] = 1
    pred = pred * mask
    s = pred.sum()
    if s < 1e-8:
        mask /= mask.sum()
        return mask
    return pred / s


SEQ_LEN = 128

def improvise(seed_tokens_original, length=200, temperature=1.0):

    ids = [vocab[t] for t in seed_tokens_original]
    types = [get_token_type(t) for t in seed_tokens_original]

    for _ in range(length):

        inp = np.array(ids[-(SEQ_LEN-1):], dtype=np.int32)
        typ = np.array(types[-(SEQ_LEN-1):], dtype=np.int32)

        # pad if too short
        if len(inp) < SEQ_LEN-1:
            pad = (SEQ_LEN-1) - len(inp)
            inp = np.concatenate([np.zeros(pad, dtype=np.int32), inp])
            typ = np.concatenate([np.zeros(pad, dtype=np.int32), typ])

        pred = model.predict([inp[None], typ[None]], verbose=0)[0]

        last_name = list(TOKEN_TYPES.keys())[types[-1]]
        pred = filter_illegal(pred, last_name)

        # temperature sampling
        pred = np.log(pred + 1e-8) / temperature
        pred = np.exp(pred)
        pred /= pred.sum()

        nxt = np.random.choice(VOCAB, p=pred)
        ids.append(nxt)
        types.append(get_token_type(rev_vocab[nxt]))

    return [rev_vocab[i] for i in ids]


# -----------------------
# MIDI → Seed → Generate
# -----------------------
def midi_to_seed(midi_path, max_tokens=128):
    tokens = midi_to_remi(midi_path)
    return tokens[:max_tokens]


if __name__ == "__main__":

    seed = midi_to_seed("./data/maestro-v3.0.0/2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi", max_tokens=128)
    print("Seed tokens loaded:", len(seed))

    seq = improvise(seed, length=300)
    remi_to_midi(seq, "jam_out.mid")

    print("🎵 Done! Saved to jam_out.mid")
