import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import json
from music21 import converter, analysis, note, chord

def load_midi(path):
    try:
        midi = pretty_midi.PrettyMIDI(path)
        notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        return notes
    except:
        print("Error reading:", path)
        return []

def note_density(notes):
    if not notes:
        return 0
    duration = notes[-1].end - notes[0].start
    if duration <= 0:
        return 0
    return len(notes) / duration

def pitch_range(notes):
    if not notes:
        return 0
    pitches = [n.pitch for n in notes]
    return max(pitches) - min(pitches)

def rhythm_variance(notes):
    if not notes:
        return 0
    durations = [n.end - n.start for n in notes]
    return np.var(durations)


#  Run evaluation for a file
def evaluate_midi(path):
    notes = load_midi(path)

    return {
        "note_density": note_density(notes),
        "pitch_range": pitch_range(notes),
        "rhythm_variance": rhythm_variance(notes),
    }

# Evaluate 3 models
if __name__ == "__main__":
    files = {
        "Transformer": "remi_transformer_jam_out.mid",
        "MusicTransformer": "remi_music_transformer_jam_output.mid",
        "MusicTransformer_Large": "remi_music_transformer_large_jam_output.mid"
    }

    results = {}

    for model_name, path in files.items():
        print("Evaluating:", model_name)
        metrics = evaluate_midi(path)
        results[model_name] = metrics
        print(metrics)


    metric_names = list(next(iter(results.values())).keys())

    data = {m: [] for m in metric_names}

    for metric in metric_names:
        for model in results:
            data[metric].append(results[model][metric])

    models = list(results.keys())
    
    def plot_bar_charts():
        for metric in metric_names:
            plt.figure(figsize=(8, 4))
            plt.title(f"{metric} Comparison")
            plt.bar(models, data[metric])
            plt.ylabel(metric)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.savefig(f"plot_{metric}.png", dpi=200, bbox_inches="tight")
            plt.show()
            plt.close()

        print("Bar charts saved (plot_<metric>.png)")

    plot_bar_charts()
