# train_remi_transformer.py
import glob
import pickle
import numpy as np
import tensorflow as tf
from tokenizer_remi import midi_to_remi, get_token_type
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

# 1. Load MIDI to REMI tokens
midi_files = glob.glob("./data/maestro-v3.0.0/**/*.midi", recursive=True)

all_tokens = []
for path in midi_files[:1000]:#[:50]:
    try:
        toks = midi_to_remi(path)
        all_tokens.append(toks)
    except:
        pass

print("Loaded tracks:", len(all_tokens))

# 2. Build vocabulary
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

# 3. Build training dataset
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

# Baseline setting
# LAYER = 4
# D_MODEL = 128
# NUM_HEADS = 8
# FF_DIM = 1024

# Large Music transformer
LAYER = 6
D_MODEL = 256
NUM_HEADS = 8
FF_DIM = 2048
MAX_LEN = SEQ_LEN

# 4. Transformer model

# Relative positional embedding
class RelativePositionEmbedding(Layer):
    def __init__(self, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.dim = dim

    def build(self, _):
        self.emb = self.add_weight(
            shape=(2*self.max_len - 1, self.dim),
            initializer="random_normal",
            trainable=True,
            name="rel_pos_emb"
        )

    def call(self, qlen, klen):
        # indices: [qlen, klen]
        range_q = tf.range(qlen)[:, None]
        range_k = tf.range(klen)[None, :]
        relative_pos = range_k - range_q + self.max_len - 1
        return tf.gather(self.emb, relative_pos)


# Multi-head with relative attention
class RelativeMultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = Dense(d_model)
        self.Wk = Dense(d_model)
        self.Wv = Dense(d_model)
        self.out = Dense(d_model)

        self.rel_emb = RelativePositionEmbedding(MAX_LEN, self.depth)

    def split_heads(self, x):
        B, L = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (B, L, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])  # (B, H, L, depth)

    def call(self, queries, keys, values):
        L = tf.shape(queries)[1]

        Q = self.split_heads(self.Wq(queries))   # (B,H,L,D)
        K = self.split_heads(self.Wk(keys))      # (B,H,L,D)
        V = self.split_heads(self.Wv(values))    # (B,H,L,D)

        # -------- Relative positions (correct) --------
        rel = self.rel_emb(L, L)                # (L,L,depth)
        rel = tf.transpose(rel, [2, 0, 1])      # (depth, L, L)

        # -------- Attention score --------
        score = tf.matmul(Q, K, transpose_b=True)   # (B,H,L,L)

        # relative logits
        score += tf.einsum("bhld,dls->bhls", Q, rel)

        score = score / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)

        out = tf.matmul(weights, V)            # (B,H,L,D)
        out = tf.transpose(out, [0,2,1,3])     # (B,L,H,D)
        out = tf.reshape(out, (tf.shape(out)[0], L, self.d_model))

        return self.out(out)


# Transformer block
def music_transformer_block(x):
    attn = RelativeMultiHeadAttention(D_MODEL, NUM_HEADS)(x, x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(FF_DIM, activation="relu")(x)
    ff = Dense(D_MODEL)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# ---------- Build full model ----------
token_in = Input(shape=(SEQ_LEN-1,), dtype="int32")
type_in  = Input(shape=(SEQ_LEN-1,), dtype="int32")

token_emb = Embedding(VOCAB, D_MODEL)(token_in)
type_emb  = Embedding(5, D_MODEL)(type_in)

x = token_emb + type_emb

for _ in range(LAYER):  # 4 layers as our version   # 6 layers same as Music Transformer
    x = music_transformer_block(x)

x = x[:, -1, :]
output = Dense(VOCAB, activation="softmax")(x)

model = Model([token_in, type_in], output)
model.compile(loss="sparse_categorical_crossentropy", optimizer = Adam(1e-4, clipnorm=1.0))

model.summary()

# 5. Train (only if run directly)
if __name__ == "__main__":

    import os
    from tensorflow.keras.models import load_model

    MODEL_PATH = "remi_music_transformer_large.h5"
    custom_objects = {
    "RelativeMultiHeadAttention": RelativeMultiHeadAttention,
}
    # Load or Initialize model
    if os.path.exists(MODEL_PATH):
        print("🔁 Found existing model — loading for continued training...")
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
    else:
        print("🆕 No existing model — training from scratch...")

    # Continue Training
    model.fit([X_tokens, X_types], Y, epochs=1, batch_size=32)

    # Save updated model
    model.save(MODEL_PATH)
    print("💾 Training complete & model saved!")
