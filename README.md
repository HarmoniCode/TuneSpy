
# TuneSpy
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

![alt text](./Styles/image.png)

TuneSpy shares similarities with the popular **Shazam** app, as both are designed to identify and match audio clips with songs from a database. While Shazam primarily focuses on real-time audio recognition using advanced fingerprinting algorithms optimized for mobile environments, TuneSpy is a desktop application aimed at exploring the core concepts of audio processing and music matching.

TuneSpy is a Python application that allows users to load audio files, generate spectrograms, extract MFCC features, and compare the loaded audio with a preprocessed database of songs to find the most similar match.




## Features

- Load audio files in various formats (MP3, WAV, FLAC)
- Generate spectrograms and save them as PNG images
- Extract MFCC features and save them as JSON files
- Hash spectrogram images using perceptual hashing
- Compare loaded audio with a preprocessed database of songs
- Display the most similar songs with similarity percentages
- Mix two audio files with adjustable weights
- Play and stop audio playback

## Video
https://drive.google.com/file/d/1ryWmfJg-txpcvSwQyOTFaXqKX9mjTL75/view?usp=sharing


## Requirements

- Python 3.x
- Required Python packages (install using `pip`):
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `imagehash`
  - `Pillow`
  - `PyQt5`
  - `soundfile`
  - `sounddevice`
  - `scipy`
  - `mutagen`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/HarmoniCode/TuneSpy.git
   cd TuneSpy
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

```sh
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
