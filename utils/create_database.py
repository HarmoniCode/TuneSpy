import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Song  # Assuming Song class is defined in main.py

def process_all_songs_in_directory(directory):
    """
    Process all songs in the directory
    args:
        directory: str, directory containing the songs
    """
    for file_name in os.listdir(directory):
        if file_name.endswith((".mp3", ".wav", ".flac")):
            file_path = os.path.join(directory, file_name)

            song = Song(file_path)
            y, sr = song.load_audio()
            S_dB = song.generate_spectrogram(y, sr)
            song.save_spectrogram(S_dB, sr)
            mfccs_list, mel_spec_list = song.extract_features(S_dB, y, sr)
            song.save_features(mfccs_list, mel_spec_list)
            mfcc_hash, mel_spec_hash = song.hash_features(mfccs_list, mel_spec_list)
            song.save_hashes(mfcc_hash, mel_spec_hash)

if __name__ == "__main__":
    directory = "./Data/Songs"
    process_all_songs_in_directory(directory)