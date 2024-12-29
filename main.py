import sys
import os
import librosa
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa.display
import imagehash
from PIL import Image
import hashlib
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='logging.log', filemode='w')
logger = logging.getLogger()

class Song:
    def __init__(self, file_path):
        self.file_path = file_path
        self.spectrogram_path = None
        self.features_path = None
        self.hash_path = None

    def load_audio(self, duration=30):
        y, sr = librosa.load(self.file_path, sr=None, duration=duration)
        logger.info(f"Loaded audio for {self.file_path}")
        return y, sr

    def generate_spectrogram(self, y, sr):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB  

    def save_spectrogram(self, S_dB, sr):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {os.path.basename(self.file_path)}')

        output_dir = "./Data/Spectrograms"
        os.makedirs(output_dir, exist_ok=True)
        self.spectrogram_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.file_path))[0] + "_spectrogram.png")
        plt.savefig(self.spectrogram_path)
        plt.close()
        logger.info(f"Spectrogram saved: {self.spectrogram_path}")

    def extract_features(self, S_dB, y, sr):
        
        mfccs = librosa.feature.mfcc(S=S_dB, sr=sr, n_mfcc=13)
        mfccs_list = mfccs.tolist()
        mel_spec_list = S_dB.tolist()
        return mfccs_list, mel_spec_list

    def save_features(self, mfccs_list, mel_spec_list):
        output_dir = "./Data/Features"
        os.makedirs(output_dir, exist_ok=True)
        self.features_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.file_path))[0] + "_features.json")
        with open(self.features_path, 'w') as json_file:
            json.dump({"mfccs": mfccs_list, "melSpec": mel_spec_list}, json_file)
        logger.info(f"Features saved: {self.features_path}")

    def hash_spectrogram_image(self):
        img = Image.open(self.spectrogram_path)
        hash_value = imagehash.phash(img)
        return str(hash_value)

    def hash_features(self, mfccs_list, mel_spec_list):
        mfccs_str = json.dumps(mfccs_list)
        mel_spec_str = json.dumps(mel_spec_list)
        mfcc_hash = hashlib.md5(mfccs_str.encode()).hexdigest()
        mel_spec_hash = hashlib.md5(mel_spec_str.encode()).hexdigest()
        return mfcc_hash, mel_spec_hash

    def save_hashes(self, spectrogram_hash, mfcc_hash, mel_spec_hash):
        output_dir = "./Data/Hashes"
        os.makedirs(output_dir, exist_ok=True)
        self.hash_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.file_path))[0] + "_hash.json")
        with open(self.hash_path, 'w') as json_file:
            json.dump({"spectrogram_hash": spectrogram_hash, "mfcc_hash": mfcc_hash, "mel_spec_hash": mel_spec_hash}, json_file)
        logger.info(f"Hashes saved: {self.hash_path}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram & Feature Extractor")
        self.setGeometry(100, 100, 600, 400)
        
        self.layout = QVBoxLayout()
        
        self.label = QLabel("No file selected")
        self.layout.addWidget(self.label)
        
        self.load_button = QPushButton("Load Song")
        self.load_button.clicked.connect(self.load_song)
        self.layout.addWidget(self.load_button)
        
        self.generate_button = QPushButton("Generate Spectrogram & Extract Features")
        self.generate_button.clicked.connect(self.generate_spectrogram_and_extract_features)
        self.generate_button.setEnabled(False)  
        self.layout.addWidget(self.generate_button)

        self.compare_button = QPushButton("Compare with Database")
        self.compare_button.clicked.connect(self.compare_with_database)
        self.compare_button.setEnabled(False)
        self.layout.addWidget(self.compare_button)

        self.results_table = QTableWidget()
        self.layout.addWidget(self.results_table)

        self.setLayout(self.layout)
        self.song = None
        self.database = self.load_database()

    def load_song(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac)")
        if file_path:
            self.song = Song(file_path)
            self.label.setText(f"Selected: {os.path.basename(file_path)}")
            self.generate_button.setEnabled(True)  
            logger.info(f"Loaded song: {file_path}")

    def generate_spectrogram_and_extract_features(self):
        if not self.song:
            return
        
        # y, sr = self.song.load_audio()
        # S_dB = self.song.generate_spectrogram(y, sr)
        # mfccs_list, mel_spec_list = self.song.extract_features(S_dB, y, sr)
        
        self.song.features_path = os.path.join("./Data/Features", os.path.splitext(os.path.basename(self.song.file_path))[0] + "_features.json")
        self.song.hash_path = os.path.join("./Data/Hashes", os.path.splitext(os.path.basename(self.song.file_path))[0] + "_hash.json")
        
        self.label.setText(f"Processing completed for {os.path.basename(self.song.file_path)}")
        self.compare_button.setEnabled(True)

    def load_database(self):
        database = {}
        features_dir = "./Data/Features"
        hashes_dir = "./Data/Hashes"
        
        for file_name in os.listdir(features_dir):
            if file_name.endswith("_features.json"):
                with open(os.path.join(features_dir, file_name), 'r') as json_file:
                    database[file_name] = json.load(json_file)
        
        for file_name in os.listdir(hashes_dir):
            if file_name.endswith("_hash.json"):
                with open(os.path.join(hashes_dir, file_name), 'r') as json_file:
                    database[file_name] = json.load(json_file)
        
        logger.info("Loaded database")
        return database

    def compare_with_database(self):
        if not self.song:
            return
        
        y, sr = self.song.load_audio()
        S_dB = self.song.generate_spectrogram(y, sr)
        self.song.save_spectrogram(S_dB, sr)  
        song_mfcc, song_mel_spec = self.song.extract_features(S_dB, y, sr)
        song_spectrogram_hash = self.song.hash_spectrogram_image()
        song_mfcc_hash, song_mel_spec_hash = self.song.hash_features(song_mfcc, song_mel_spec)
        
        similarities = []
        
        weight_mfcc = 0.4
        weight_mel_spec = 0.4
        weight_hash = 0.2
        
        for file_name, data in self.database.items():
            if file_name.endswith("_features.json"):
                
                hash_file_name = file_name.replace("_features.json", "_hash.json")
                with open(os.path.join("./Data/Hashes", hash_file_name), 'r') as hash_file:
                    db_hashes = json.load(hash_file)
                    db_spectrogram_hash = db_hashes["spectrogram_hash"]
                    db_mfcc_hash = db_hashes["mfcc_hash"]
                    db_mel_spec_hash = db_hashes["mel_spec_hash"]
                    spectrogram_hash_similarity = self.calculate_hash_similarity(song_spectrogram_hash, db_spectrogram_hash)
                    mfcc_hash_similarity = self.calculate_md5_hash_similarity(song_mfcc_hash, db_mfcc_hash)
                    mel_spec_hash_similarity = self.calculate_md5_hash_similarity(song_mel_spec_hash, db_mel_spec_hash)
                
                
                similarity = (weight_mfcc * mfcc_hash_similarity + weight_mel_spec * mel_spec_hash_similarity + 100*weight_hash * spectrogram_hash_similarity)
                
                print(f"File: {file_name}, MFCC Hash Similarity: {mfcc_hash_similarity:.2f}%, MelSpec Hash Similarity: {mel_spec_hash_similarity:.2f}%, Spectrogram Hash Similarity: {spectrogram_hash_similarity:.2f}")
                similarities.append((file_name, similarity))
    
        similarities.sort(key=lambda x: x[1], reverse=True)
    
        self.results_table.setRowCount(len(similarities))
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Song", "Similarity"])
    
        for i, (file_name, similarity) in enumerate(similarities):
            self.results_table.setItem(i, 0, QTableWidgetItem(file_name))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{similarity:.2f}%"))
    
        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()
        self.results_table.show()

    def calculate_hash_similarity(self, hash1_str, hash2_str):
        hash1 = imagehash.hex_to_hash(hash1_str)
        hash2 = imagehash.hex_to_hash(hash2_str)
        return max(0, 1 - (hash1 - hash2) / len(hash1.hash))  

    def calculate_md5_hash_similarity(self, hash1_str, hash2_str):
        
        hash1_bin = bin(int(hash1_str, 16))[2:].zfill(128)
        hash2_bin = bin(int(hash2_str, 16))[2:].zfill(128)
        
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1_bin, hash2_bin))
        
        similarity = 1 - (hamming_distance / 128)
        return similarity * 100  

    def normalize_features(self, features):
        features = np.array(features)
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / std

    def calculate_similarity(self, feature1, feature2):
        feature1 = self.normalize_features(feature1)
        feature2 = self.normalize_features(feature2)
        
        min_length = min(feature1.shape[1], feature2.shape[1])
        feature1 = feature1[:, :min_length]
        feature2 = feature2[:, :min_length]
        
        dot_product = np.dot(feature1.flatten(), feature2.flatten())
        norm1 = np.linalg.norm(feature1.flatten())
        norm2 = np.linalg.norm(feature2.flatten())
        similarity = dot_product / (norm1 * norm2)
        return abs(similarity) * 100  

def process_all_songs_in_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(('.mp3', '.wav', '.flac')):
            file_path = os.path.join(directory, file_name)
            logger.info(f"Processing {file_path}")
            
            song = Song(file_path)
            y, sr = song.load_audio()
            S_dB = song.generate_spectrogram(y, sr)
            song.save_spectrogram(S_dB, sr)
            mfccs_list, mel_spec_list = song.extract_features(S_dB, y, sr)
            song.save_features(mfccs_list, mel_spec_list)
            spectrogram_hash = song.hash_spectrogram_image()
            mfcc_hash, mel_spec_hash = song.hash_features(mfccs_list, mel_spec_list)
            song.save_hashes(spectrogram_hash, mfcc_hash, mel_spec_hash)
            logger.info(f"Processed {file_path}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Uncomment the following line to process all songs in the directory
    process_all_songs_in_directory("./Data/Songs")
    # main()