import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

SEQUENCE_FILE = "data/maestro_sequences.pkl"

# ---- Hyperparameters ----
SEQ_LEN = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3

# ---- Model ----
class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out)

# ---- Load data ----
with open(SEQUENCE_FILE, "rb") as f:
    sequences = pickle.load(f)

notes = [p for seq in sequences for (p, _) in seq]
notes = np.array(notes)

# Create vocab
pitches = sorted(set(notes))
pitch2idx = {p: i for i, p in enumerate(pitches)}
idx2pitch = {i: p for p, i in pitch2idx.items()}
vocab_size = len(pitches)

# Convert to integer indices
encoded = np.array([pitch2idx[p] for p in notes])

# Split into sequences
inputs, targets = [], []
for i in range(len(encoded) - SEQ_LEN):
    inputs.append(encoded[i:i+SEQ_LEN])
    targets.append(encoded[i+1:i+SEQ_LEN+1])
inputs = np.array(inputs)
targets = np.array(targets)

# ---- Dataloader ----
tensor_x = torch.tensor(inputs, dtype=torch.long)
tensor_y = torch.tensor(targets, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Train ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MelodyLSTM(vocab_size, HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

torch.save({"model": model.state_dict(), "idx2pitch": idx2pitch, "pitch2idx": pitch2idx}, "lstm_model.pt")
print("✅ Model saved to lstm_model.pt")
