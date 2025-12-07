# AI-Assisted-Musical-Improvisation-System
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
