# tokenizer_remi.py
import pretty_midi
import numpy as np

BAR = "Bar"
POS = "Pos_"
PITCH = "Pitch_"
DUR = "Dur_"
VEL = "Vel_"

GRID = 16
MAX_DUR = 64
MAX_VEL = 32

TOKEN_TYPES = {
    "Bar": 0,
    "Pos": 1,
    "Pitch": 2,
    "Dur": 3,
    "Vel": 4
}

def get_token_type(token):
    if token == "Bar":
        return TOKEN_TYPES["Bar"]
    prefix = token.split("_")[0]
    return TOKEN_TYPES[prefix]

def midi_to_remi(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    if not notes:
        return []

    notes.sort(key=lambda n: n.start)
    tempo = midi.estimate_tempo()
    beat = 60 / tempo

    tokens = []
    current_bar = -1

    for n in notes:
        bar = int(n.start / (beat * 4))
        pos = int((n.start % (beat * 4)) / ((beat * 4) / GRID))

        if bar != current_bar:
            tokens.append(BAR)
            current_bar = bar

        tokens.append(f"{POS}{pos}")
        tokens.append(f"{PITCH}{n.pitch}")

        dur = n.end - n.start
        dur_q = min(int(dur / (beat / GRID)), MAX_DUR - 1)
        tokens.append(f"{DUR}{dur_q}")

        vel_q = int(n.velocity * MAX_VEL / 128)
        tokens.append(f"{VEL}{vel_q}")

    return tokens

def remi_to_midi(tokens, out_path="remi_out.mid"):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    tempo = 120
    beat = 60 / tempo

    bar_t = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == BAR:
            bar_t += beat * 4
            i += 1
            continue

        if tok.startswith(POS):
            if i + 3 >= len(tokens):
                break

            pos = int(tok.split("_")[1])

            if not tokens[i+1].startswith(PITCH):
                i += 1; continue
            if not tokens[i+2].startswith(DUR):
                i += 1; continue
            if not tokens[i+3].startswith(VEL):
                i += 1; continue

            pitch = int(tokens[i+1].split("_")[1])
            dur = int(tokens[i+2].split("_")[1]) * (beat / GRID)
            vel = int(tokens[i+3].split("_")[1]) * (128 / MAX_VEL)

            t = bar_t + pos * (beat * 4 / GRID)

            inst.notes.append(pretty_midi.Note(
                pitch=pitch,
                velocity=int(vel),
                start=t,
                end=t + dur
            ))

            i += 4
            continue

        i += 1

    midi.instruments.append(inst)
    midi.write(out_path)
    print("Saved:", out_path)
