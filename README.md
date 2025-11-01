# AI-Assisted-Musical-Improvisation-System

### Project Plan
Phase 1 – Baseline: Supervised Music Continuation

Train or fine-tune a Transformer/LSTM on symbolic music data (like MAESTRO or Lakh MIDI).

Input: a short melody segment (4–8 bars).

Output: next-segment prediction (AI continuation).

This already allows "offline jamming" by alternating human-AI phrases.

Phase 2 – RL Fine-tuning (Stretch Goal)

Use reinforcement learning to reward musicality:

Reward 1: tonal consistency (same key / harmonic compatibility).

Reward 2: rhythmic variation (not too repetitive).

Reward 3: novelty (new notes not in input).

Use something like REINFORCE or policy gradient on the generated MIDI tokens.

### Setup github
…or create a new repository on the command line
```bash
echo "# AI-Assisted-Musical-Improvisation-System-" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/bcchang83/AI-Assisted-Musical-Improvisation-System-.git
git push -u origin main
```

…or push an existing repository from the command line
```bash
git remote add origin https://github.com/bcchang83/AI-Assisted-Musical-Improvisation-System-.git
git branch -M main
git push -u origin main
```
