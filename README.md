# AI-Assisted-Musical-Improvisation-System
This project introduces the AI-Assisted Musical Improvisation System (AIMIS), a symbolic call-and-response framework for music generation. Using the Maestro dataset and REMI preprocessing, we benchmarked a standard Transformer against the Music Transformer architecture. Our experiments demonstrate that the Music Transformer outperforms the baseline in rhythmic coherence and phrasing. Furthermore, scaling to a larger model capacity improves quantitative metrics such as pitch range and rhythm variance, indicating greater expressive potential in the generated music.
### Dataset

The MAESTRO Dataset (v3)

https://magenta.tensorflow.org/datasets/maestro#download

### Setup Instructions (Environment Setup)

1. Clone this repository:
   ```bash
   git clone https://github.com/bcchang83/AI-Assisted-Musical-Improvisation-System-.git
2. Create and open a virtual environment:
   ```bash
   conda create --name API python=3.9
   conda activate API
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt


### How to use it

1. Train a model or download a trained model from (https://drive.google.com/drive/folders/1HcNuf_Zqzxw5N39t3DH04fDYseMxJe5G?usp=drive_link)
   ```bash
   python train_remi_music_transformer.py
2. Generate music
   ```bash
   python generate_remi_music_transformer.py

### Code Explaination
1. train_remi_transformer.py : Train the vanilla Transformer with REMI tokens.
2. train_remi_music_transformer.py : Train Music Transformer with REMI tokens.
3. generate_remi_transformer.py : Generate music by the vanilla Transformer.
4. generate_remi_music_transformer.py : Generate music by Music Transformer.
5. tokenizer_remi.py : Preprocessing with REMI tokens.
6. evaluation.py : Generate evaluation figures.

### Demo
Those reults use the same input clip from "maestro-v3.0.0/2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi".

Input clip: https://drive.google.com/file/d/1rWBvJGfgKcF0FW6O09MHqpl24B6RCqz5/view?usp=drive_link

Standard Transformer: https://drive.google.com/file/d/18LnvZIDIPDCwH-dUzmKW9wtT1ZEHQdVD/view?usp=drive_link

Music Transformer: https://drive.google.com/file/d/1MtD0fK-WJnXYIJOMQQdPBV-CpbKMgTwe/view?usp=sharing

Music Transformer (Large): https://drive.google.com/file/d/1x0IumGznesstCQ2Ekmjn_Upuma1tIhll/view?usp=drive_link

Mixed input with Music Transformer (Large) output: https://drive.google.com/file/d/1TBuhbrToMwtXfhOK7wi6VuSW8S6ol2Lc/view?usp=drive_link
