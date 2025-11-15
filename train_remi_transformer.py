# train_remi_transformer.py
import glob
import pickle
import numpy as np
from tokenizer_remi import midi_to_remi, get_token_type
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==============================
# 1. Load MIDI → REMI tokens
# ==============================
midi_files = glob.glob("./data/maestro-v3.0.0/**/*.midi", recursive=True)

all_tokens = []
for path in midi_files[:50]:
    try:
        toks = midi_to_remi(path)
        all_tokens.append(toks)
    except:
        pass

print("Loaded tracks:", len(all_tokens))

# ==============================
# 2. Build vocabulary
# ==============================
vocab = {}
for seq in all_tokens:
    for t in seq:
        if t not in vocab:
            vocab[t] = len(vocab)

rev_vocab = {i: t for t, i in vocab.items()}
VOCAB = len(vocab)

pickle.dump(vocab, open("vocab.pkl", "wb"))
pickle.dump(rev_vocab, open("rev_vocab.pkl", "wb"))

print("Vocab size:", VOCAB)

# ==============================
# 3. Build training dataset
# ==============================
SEQ_LEN = 128

X_tokens = []
X_types = []
Y = []

for seq in all_tokens:
    ids = [vocab[t] for t in seq]
    types = [get_token_type(t) for t in seq]
    if len(ids) < SEQ_LEN:
        continue

    for i in range(len(ids) - SEQ_LEN):
        X_tokens.append(ids[i:i+SEQ_LEN-1])
        X_types.append(types[i:i+SEQ_LEN-1])
        Y.append(ids[i+SEQ_LEN-1])

X_tokens = np.array(X_tokens, dtype=np.int32)
X_types  = np.array(X_types, dtype=np.int32)
Y        = np.array(Y, dtype=np.int32)

print("Dataset:", X_tokens.shape, Y.shape)

TYPE_VOCAB = 5
D_MODEL = 128


# ==============================
# 4. Transformer model
# ==============================
def transformer_block(x):
    attn = MultiHeadAttention(num_heads=4, key_dim=D_MODEL)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    ff = Dense(D_MODEL*4, activation="relu")(x)
    ff = Dense(D_MODEL)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    return x


token_in = Input(shape=(SEQ_LEN-1,), dtype="int32")
type_in  = Input(shape=(SEQ_LEN-1,), dtype="int32")

token_emb = Embedding(VOCAB, D_MODEL)(token_in)
type_emb  = Embedding(TYPE_VOCAB, D_MODEL)(type_in)

x = Add()([token_emb, type_emb])

for _ in range(4):
    x = transformer_block(x)

x = x[:, -1, :]  # predict next token
output = Dense(VOCAB, activation="softmax")(x)

model = Model([token_in, type_in], output)
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(1e-4))
model.summary()

# ===========================================
# 5. Train (only if run directly)
# ===========================================
if __name__ == "__main__":
    model.fit([X_tokens, X_types], Y, epochs=10, batch_size=32)
    model.save("remi_transformer.h5")
    print("Training complete!")
