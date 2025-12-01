import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import json
from music21 import converter, analysis, note, chord

# ============================
#  Helper: load MIDI
# ============================
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

# ============================
#   1. Note Density (notes/sec)
# ============================
def note_density(notes):
    if not notes:
        return 0
    duration = notes[-1].end - notes[0].start
    if duration <= 0:
        return 0
    return len(notes) / duration

# ============================
#   2. Pitch Range
# ============================
def pitch_range(notes):
    if not notes:
        return 0
    pitches = [n.pitch for n in notes]
    return max(pitches) - min(pitches)

# ============================
#   3. Pitch Entropy
# ============================
def pitch_entropy(notes):
    if not notes:
        return 0
    pitches = [n.pitch for n in notes]
    values, counts = np.unique(pitches, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

# ============================
#   4. Rhythm stability (duration variance)
# ============================
def rhythm_variance(notes):
    if not notes:
        return 0
    durations = [n.end - n.start for n in notes]
    return np.var(durations)

# ============================
#   5. Polyphony Level
# ============================
def polyphony(notes):
    if not notes:
        return 0
    # sample 1000 time points
    starts = np.linspace(notes[0].start, notes[-1].end, 1000)
    counts = []
    for t in starts:
        c = 0
        for n in notes:
            if n.start <= t <= n.end:
                c += 1
        counts.append(c)
    return np.mean(counts)

# ============================
#   6. Tonal stability (Music21 key analysis)
# ============================
def tonal_stability(path):
    try:
        score = converter.parse(path)
        key = score.analyze("key")
        return key.correlationCoefficient  # 0~1, high=stable tonality
    except:
        return 0

# ============================
#  Run evaluation for a file
# ============================
def evaluate_midi(path):
    notes = load_midi(path)

    return {
        "note_density": note_density(notes),
        "pitch_range": pitch_range(notes),
        "pitch_entropy": pitch_entropy(notes),
        "rhythm_variance": rhythm_variance(notes),
        "polyphony": polyphony(notes),
        "tonal_stability": tonal_stability(path)
    }

# ============================
#  Example: Evaluate 3 models
# ============================
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

    # --------------------------
    # Bar chart (each metric)
    # --------------------------
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

    # --------------------------
    # Radar chart (all metric)
    # --------------------------
    def plot_radar():
        num_metrics = len(metric_names)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for i, model in enumerate(models):
            vals = [results[model][m] for m in metric_names]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=model)
            ax.fill(angles, vals, alpha=0.15)

        ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.title("Model Comparison Radar Chart")
        plt.savefig("plot_radar.png", dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

        print("Radar chart saved (plot_radar.png)")

    # --------------------------
    # RUN
    # --------------------------
    plot_bar_charts()
    plot_radar()

    print("All evaluation plots generated!")