import sys
import os
import librosa
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa.display
import imagehash
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='logging.log', filemode='w')
logger = logging.getLogger()

class Song:
    def __init__(self, file_path):
        self.file_path = file_path
        self.spectrogram_path = None
        self.mfcc_path = None
        self.hash_path = None

    def load_audio(self, duration=30):
        y, sr = librosa.load(self.file_path, sr=None, duration=duration)
        logger.info(f"Loaded audio for {self.file_path}")
        return y, sr

    def generate_spectrogram(self, y, sr):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

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
        return self.spectrogram_path

    def extract_features(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_list = mfccs.tolist()

        output_dir = "./Data/Features"
        os.makedirs(output_dir, exist_ok=True)
        self.mfcc_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.file_path))[0] + "_mfcc.json")
        with open(self.mfcc_path, 'w') as json_file:
            json.dump({"mfccs": mfccs_list}, json_file)
        logger.info(f"MFCC features saved: {self.mfcc_path}")
        return self.mfcc_path

    def hash_spectrogram_image(self):
        img = Image.open(self.spectrogram_path)
        hash_value = imagehash.phash(img)

        output_dir = "./Data/Hashes"
        os.makedirs(output_dir, exist_ok=True)
        self.hash_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.file_path))[0] + "_hash.json")
        with open(self.hash_path, 'w') as json_file:
            json.dump({"image_hash": str(hash_value)}, json_file)
        logger.info(f"Spectrogram hash saved: {self.hash_path}")
        return self.hash_path

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
        
        y, sr = self.song.load_audio()
        self.song.generate_spectrogram(y, sr)
        self.song.extract_features(y, sr)
        self.song.hash_spectrogram_image()
        self.label.setText(f"Processing completed for {os.path.basename(self.song.file_path)}")
        self.compare_button.setEnabled(True)

    def load_database(self):
        database = {}
        features_dir = "./Data/Features"
        hashes_dir = "./Data/Hashes"
        
        for file_name in os.listdir(features_dir):
            if file_name.endswith("_mfcc.json"):
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
        
        song_mfcc_path = self.song.mfcc_path
        song_hash_path = self.song.hash_path
        
        with open(song_mfcc_path, 'r') as json_file:
            song_mfcc = json.load(json_file)["mfccs"]
        
        with open(song_hash_path, 'r') as json_file:
            song_hash = json.load(json_file)["image_hash"]
        
        similarities = []
        
        for file_name, data in self.database.items():
            if file_name.endswith("_mfcc.json"):
                db_mfcc = data["mfccs"]
                similarity = self.calculate_similarity(song_mfcc, db_mfcc)
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

    def calculate_similarity(self, mfcc1, mfcc2):
        mfcc1 = np.array(mfcc1)
        mfcc2 = np.array(mfcc2)
        
        min_length = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_length]
        mfcc2 = mfcc2[:, :min_length]
        
        dot_product = np.dot(mfcc1.flatten(), mfcc2.flatten())
        norm1 = np.linalg.norm(mfcc1.flatten())
        norm2 = np.linalg.norm(mfcc2.flatten())
        similarity = dot_product / (norm1 * norm2)
        return similarity * 100

def process_all_songs_in_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(('.mp3', '.wav', '.flac')):
            file_path = os.path.join(directory, file_name)
            logger.info(f"Processing {file_path}")
            
            song = Song(file_path)
            y, sr = song.load_audio()
            song.generate_spectrogram(y, sr)
            song.extract_features(y, sr)
            song.hash_spectrogram_image()
            logger.info(f"Processed {file_path}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Uncomment the following line to process all songs in a directory before run
    # process_all_songs_in_directory("./Data/Songs")
    main()