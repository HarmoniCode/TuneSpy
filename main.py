import sys
import os
import librosa
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa.display
import imagehash
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QSlider,
    QFrame
)
from PyQt5.QtGui import QPixmap
import soundfile as sf
from scipy.signal import resample
import logging
import sounddevice as sd
import mutagen

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logging.log",
    filemode="w",
)
logger = logging.getLogger()


class Song:
    def __init__(self, file_path):
        # Initialize the Song object with the file path and other attributes
        self.file_path = file_path
        self.spectrogram_path = None
        self.features_path = None
        self.hash_path = None
        self.duration = 30  # Duration to load from the audio file (in seconds)
        self.cover_image = None  # Attribute to store the cover image

    def load_audio(self):
        """
        Load the audio file and return the time series and sample rate
        Returns:
            time (np.array): Time series of the audio
            sample_rate (int): Sample rate of the audio
        """
        time, sample_rate = librosa.load(
            self.file_path, sr=None, duration=self.duration
        )
        logger.info(f"Loaded audio for {self.file_path}")
        return time, sample_rate
    
    def load_cover_image(self):
        """
        Load the cover image from the audio file if available.
        """
        try:
            audio = mutagen.File(self.file_path)
            for tag in audio.tags.values():
                if isinstance(tag, mutagen.id3.APIC):
                    self.cover_image = tag.data
                    return
        except Exception as e:
            logger.error(f"Error loading album cover: {e}")
        self.cover_image = None

    def generate_spectrogram(self, time, sample_rate):
        """
        Generate the spectrogram of the audio file
        Args:
            time (np.array): Time series of the audio
            sample_rate (int): Sample rate of the audio
        Returns:
            spectrogram_dB (np.array): Spectrogram in decibel units
        """
        spectrogram = librosa.feature.melspectrogram(
            y=time, sr=sample_rate, n_mels=128, fmax=8000
        )
        spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_dB

    def save_spectrogram(self, spectrogram_dB, sample_rate):
        """
        Save the spectrogram as an image
        Args:
            spectrogram_dB (np.array): Spectrogram in decibel units
            sample_rate (int): Sample rate of the audio
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spectrogram_dB, x_axis="time", y_axis="mel", sr=sample_rate
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram of {os.path.basename(self.file_path)}")

        output_dir = "./Data/Spectrograms"
        os.makedirs(output_dir, exist_ok=True)
        self.spectrogram_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(self.file_path))[0] + "_spectrogram.png",
        )
        plt.savefig(self.spectrogram_path)
        plt.close()
        logger.info(f"Spectrogram saved: {self.spectrogram_path}")

    def extract_features(self, spectrogram_dB, time, sample_rate):
        """
        Extract the features from the spectrogram
        Args:
            spectrogram_dB (np.array): Spectrogram in decibel units
            time (np.array): Time series of the audio
            sample_rate (int): Sample rate of the audio
        Returns:
            mfccs_list (list): List of MFCCs
            mel_spec_list (list): List of Mel Spectrogram
        """
        mfccs = librosa.feature.mfcc(S=spectrogram_dB, sr=sample_rate, n_mfcc=13)
        mfccs_list = mfccs.tolist()
        mel_spec_list = spectrogram_dB.tolist()
        return mfccs_list, mel_spec_list

    def save_features(self, mfccs_list, mel_spec_list):
        """
        Save the features as a JSON file
        Args:
            mfccs_list (list): List of MFCCs
            mel_spec_list (list): List of Mel Spectrogram
        """
        output_dir = "./Data/Features"
        os.makedirs(output_dir, exist_ok=True)
        self.features_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(self.file_path))[0] + "_features.json",
        )
        with open(self.features_path, "w") as json_file:
            json.dump({"mfccs": mfccs_list, "melSpec": mel_spec_list}, json_file)
        logger.info(f"Features saved: {self.features_path}")

    def hash_features(self, mfccs_list, mel_spec_list):
        """
        Hash the features using pHash
        Args:
            mfccs_list (list): List of MFCCs
            mel_spec_list (list): List of Mel Spectrogram
        Returns:
            spec_hash (str): Hash of the spectrogram image
            mfcc_hash (str): Hash of the MFCCs image
            mel_spec_hash (str): Hash of the Mel Spectrogram image
        """
        temp_dir = os.path.join("./Data", "Temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Hash the spectrogram image
        spec_img = Image.open(self.spectrogram_path)
        spec_hash = imagehash.phash(spec_img)

        # Convert MFCCs to an image and hash it
        mfccs_array = np.array(mfccs_list)
        mfcc_temp_path = os.path.join(temp_dir, "mfccs_temp.png")
        plt.imshow(mfccs_array, cmap="viridis", aspect="auto")
        plt.axis("off")
        plt.savefig(mfcc_temp_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        mfcc_img = Image.open(mfcc_temp_path)
        mfcc_hash = imagehash.phash(mfcc_img)
        os.remove(mfcc_temp_path)

        # Convert Mel spectrogram to an image and hash it
        mel_spec_array = np.array(mel_spec_list)
        mel_spec_temp_path = os.path.join(temp_dir, "mel_spec_temp.png")
        plt.imshow(mel_spec_array, cmap="viridis", aspect="auto")
        plt.axis("off")
        plt.savefig(mel_spec_temp_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        mel_spec_img = Image.open(mel_spec_temp_path)
        mel_spec_hash = imagehash.phash(mel_spec_img)
        os.remove(mel_spec_temp_path)

        return str(spec_hash), str(mfcc_hash), str(mel_spec_hash)

    def save_hashes(self, spectrogram_hash, mfcc_hash, mel_spec_hash):
        """
        Save the hashes as a JSON file
        Args:
            spectrogram_hash (str): Hash of the spectrogram
            mfcc_hash (str): Hash of the MFCCs
            mel_spec_hash (str): Hash of the Mel Spectrogram
        """
        output_dir = "./Data/Hashes"
        os.makedirs(output_dir, exist_ok=True)
        self.hash_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(self.file_path))[0] + "_hash.json",
        )
        with open(self.hash_path, "w") as json_file:
            json.dump(
                {
                    "spectrogram_hash": spectrogram_hash,
                    "mfcc_hash": mfcc_hash,
                    "mel_spec_hash": mel_spec_hash,
                },
                json_file,
            )
        logger.info(f"Hashes saved: {self.hash_path}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.file1 = None
        self.file2 = None
        self.song = None
        self.setWindowTitle("Spectrogram & Feature Extractor")
        self.setGeometry(100, 100, 600, 400)

        self.layout = QHBoxLayout()

        left_frame=QFrame()
        left_frame.setFixedWidth(400)
        left_layout = QVBoxLayout()
        left_frame.setLayout(left_layout)

        right_frame=QFrame()
        right_layout = QVBoxLayout()
        right_frame.setLayout(right_layout)

        self.layout.addWidget(left_frame)
        self.layout.addWidget(right_frame)

        self.label = QLabel("No file selected")
        # right_layout.addWidget(self.label)

        image_frame = QFrame()
        # image_frame.setStyleSheet("border: 1px solid red;")
        image_layout = QVBoxLayout()
        image_layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        image_frame.setLayout(image_layout)
        self.cover_label = QLabel()
        self.cover_label.setStyleSheet("border: 1px solid black; border-radius: 20px;")
        self.cover_label.setFixedSize(800, 800)
        self.cover_label.setAlignment(Qt.AlignCenter)  # Center the image horizontally and vertically
        image_layout.addWidget(self.cover_label)
        right_layout.addWidget(image_frame)

        default_image_path = "./Styles/logo.png"
        if os.path.exists(default_image_path):
            pixmap = QPixmap(default_image_path)
            self.cover_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio))
        else:
            self.cover_label.setText("No Image")
            self.cover_label.setAlignment(Qt.AlignCenter)
            # self.cover_label.setStyleSheet("font-size: 50px; border: 1px solid black;")

        self.load_button = QPushButton("Load Song")
        self.load_button.clicked.connect(self.load_song)
        left_layout.addWidget(self.load_button)

        mix_frame=QFrame()
        mix_layout=QVBoxLayout()
        mix_frame.setLayout(mix_layout)

        self.load_mix_song_1_button = QPushButton("Load Song 1 for Mixing")
        self.load_mix_song_1_button.clicked.connect(self.load_mix_song_1)
        mix_layout.addWidget(self.load_mix_song_1_button)

        self.load_mix_song_2_button = QPushButton("Load Song 2 for Mixing")
        self.load_mix_song_2_button.clicked.connect(self.load_mix_song_2)
        mix_layout.addWidget(self.load_mix_song_2_button)

        self.mix_audio_button = QPushButton("Mix Audio")
        self.mix_audio_button.clicked.connect(self.mix_audio)
        mix_layout.addWidget(self.mix_audio_button)

        left_layout.addWidget(mix_frame)

        control_frame = QFrame()
        control_layout = QHBoxLayout()
        control_frame.setLayout(control_layout)

        self.play_mix_button = QPushButton("Play Audio")
        self.play_mix_button.clicked.connect(self.play_audio)
        control_layout.addWidget(self.play_mix_button)

        self.stop_button = QPushButton("Stop Audio")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_audio)
        control_layout.addWidget(self.stop_button)

        right_layout.addWidget(control_frame)

        H_layout1 = QHBoxLayout()
        self.slider1_label = QLabel("Weight of Audio 1")
        H_layout1.addWidget(self.slider1_label)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(50)
        H_layout1.addWidget(self.slider1)
        self.slider1.setEnabled(False)

        H_layout2 = QHBoxLayout()

        self.slider2_label = QLabel("Weight of Audio 2")
        H_layout2.addWidget(self.slider2_label)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(50)
        H_layout2.addWidget(self.slider2)
        self.slider2.setEnabled(False)

        left_layout.addLayout(H_layout1)
        left_layout.addLayout(H_layout2)

        self.results_table = QTableWidget()
        left_layout.addWidget(self.results_table)

        self.setLayout(self.layout)
        self.song = None
        self.database = self.load_database()

    def load_song(self):
        """
        Load a song file using a file dialog.
        Sets the selected song to the Song object and updates the UI.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac)"
        )
        if file_path:
            self.song = Song(file_path)
            self.song.load_cover_image()
            self.label.setText(f"Selected: {os.path.basename(file_path)}")
            self.display_cover_image()
            logger.info(f"Loaded song: {file_path}")

        self.compare_with_database()

    def display_cover_image(self):
        """
        Display the cover image in the cover_label.
        If no image is found, display the first letter of the song name.
        """
        if self.song.cover_image:
            pixmap = QPixmap()
            pixmap.loadFromData(self.song.cover_image)
            self.cover_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio))
        else:
            song_name = os.path.basename(self.song.file_path)
            first_word = song_name.split('_')[0].upper()
            self.cover_label.setText(first_word)
            self.cover_label.setAlignment(Qt.AlignCenter)
            self.cover_label.setStyleSheet("font-size: 100px;border: 1px solid black; border-radius: 20px;")

    def load_database(self):
        """
        Load the database of features and hashes from the Data directory.
        Returns:
            database (dict): Dictionary containing features and hashes of songs.
        """
        database = {}
        features_dir = "./Data/Features"
        hashes_dir = "./Data/Hashes"

        for file_name in os.listdir(features_dir):
            if file_name.endswith("_features.json"):
                with open(os.path.join(features_dir, file_name), "r") as json_file:
                    database[file_name] = json.load(json_file)

        for file_name in os.listdir(hashes_dir):
            if file_name.endswith("_hash.json"):
                with open(os.path.join(hashes_dir, file_name), "r") as json_file:
                    database[file_name] = json.load(json_file)

        logger.info("Loaded database")
        return database

    def compare_with_database(self):
        """
        Compare the loaded song with the songs in the database.
        Calculates similarity based on spectrogram, MFCC, and Mel spectrogram hashes and features.
        """
        if not self.song:
            return
        
        self.song.features_path = os.path.join(
            "./Data/Features",
            os.path.splitext(os.path.basename(self.song.file_path))[0]
            + "_features.json",
        )
        self.song.hash_path = os.path.join(
            "./Data/Hashes",
            os.path.splitext(os.path.basename(self.song.file_path))[0] + "_hash.json",
        )

        self.label.setText(
            f"Processing completed for {os.path.basename(self.song.file_path)}"
        )


        time, sample_rate = self.song.load_audio()
        spectrogram_dB = self.song.generate_spectrogram(time, sample_rate)
        self.song.save_spectrogram(spectrogram_dB, sample_rate)
        song_mfcc, song_mel_spec = self.song.extract_features(
            spectrogram_dB, time, sample_rate
        )
        song_spectrogram_hash, song_mfcc_hash, song_mel_spec_hash = (
            self.song.hash_features(song_mfcc, song_mel_spec)
        )

        similarities = []

        weight_mfcc = 0.4
        weight_mel_spec = 0.4
        weight_hash = 0.2

        for file_name, data in self.database.items():
            if file_name.endswith("_features.json"):
                hash_file_name = file_name.replace("_features.json", "_hash.json")
                with open(
                    os.path.join("./Data/Hashes", hash_file_name), "r"
                ) as hash_file:
                    db_hashes = json.load(hash_file)
                    db_spectrogram_hash = db_hashes["spectrogram_hash"]
                    db_mfcc_hash = db_hashes["mfcc_hash"]
                    db_mel_spec_hash = db_hashes["mel_spec_hash"]

                    spectrogram_hash_similarity = self.calculate_hash_similarity(
                        song_spectrogram_hash, db_spectrogram_hash
                    )
                    mfcc_hash_similarity = self.calculate_hash_similarity(
                        song_mfcc_hash, db_mfcc_hash
                    )
                    mel_spec_hash_similarity = self.calculate_hash_similarity(
                        song_mel_spec_hash, db_mel_spec_hash
                    )

                    mfcc_similarity = self.calculate_similarity(
                        song_mfcc, data["mfccs"]
                    )
                    mel_spec_similarity = self.calculate_similarity(
                        song_mel_spec, data["melSpec"]
                    )

                    # Combine hash-based and cosine similarity scores
                    hash_similarity = (
                        weight_mfcc * mfcc_hash_similarity
                        + weight_mel_spec * mel_spec_hash_similarity
                        + weight_hash * spectrogram_hash_similarity
                    )*100
                    print("Hash-based similarity: ", hash_similarity)
                    cosine_similarity = (mfcc_similarity + mel_spec_similarity) / 2
                    print("Cosine similarity: ", cosine_similarity)

                    # Normalize the combined similarity score
                    similarity = (hash_similarity + cosine_similarity) / 2

                    print(
                        f"File: {file_name}, MFCC Hash Similarity: {mfcc_hash_similarity:.2f}%, MelSpec Hash Similarity: {mel_spec_hash_similarity:.2f}%, Spectrogram Hash Similarity: {spectrogram_hash_similarity:.2f}, MFCC Cosine Similarity: {mfcc_similarity:.2f}%, MelSpec Cosine Similarity: {mel_spec_similarity:.2f}%"
                    )
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
        """
        Calculate the similarity between two perceptual hashes.
        Args:
            hash1_str (str): Hash of the first image.
            hash2_str (str): Hash of the second image.
        Returns:
            similarity (float): Similarity score between 0 and 1.
        """
        hash1 = imagehash.hex_to_hash(hash1_str)
        hash2 = imagehash.hex_to_hash(hash2_str)
        return max(0, 1 - (hash1 - hash2) / len(hash1.hash))

    def normalize_features(self, features):
        """
        Normalize the features to have zero mean and unit variance.
        Args:
            features (np.array): Features to be normalized.
        Returns:
            normalized_features (np.array): Normalized features.
        """
        features = np.array(features)
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / std

    def calculate_similarity(self, feature1, feature2):
        """
        Calculate the cosine similarity between two feature arrays.
        Args:
            feature1 (np.array): First feature array.
            feature2 (np.array): Second feature array.
        Returns:
            similarity (float): Cosine similarity score between 0 and 100.
        """
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

    def check_mix_files_loaded(self):
        """
        Check if both mix files are loaded and enable the play button if they are.
        """
        if self.file1 and self.file2:
            self.play_mix_button.setEnabled(True)
            self.slider1.setEnabled(True)
            self.slider2.setEnabled(True)
        else:
            self.play_mix_button.setEnabled(False)
            self.slider1.setEnabled(False)
            self.slider2.setEnabled(False)

    def load_mix_song_1(self):
        """
        Load the first song for mixing using a file dialog.
        Updates the UI and checks if both mix files are loaded.
        Returns:
            file1 (str): Path to the first audio file.
        """
        self.file1, _ = QFileDialog.getOpenFileName(
            self, "Select First Audio File", "", "Audio Files (*.wav *.flac *.ogg)"
        )
        if not self.file1:
            return None, None
        self.load_mix_song_1_button.setText(
            f"Load Song 1 for Mixing, Loaded File: {os.path.basename(self.file1)}"
        )
        self.check_mix_files_loaded()
        return self.file1

    def load_mix_song_2(self):
        """
        Load the second song for mixing using a file dialog.
        Updates the UI and checks if both mix files are loaded.
        Returns:
            file2 (str): Path to the second audio file.
        """
        self.file2, _ = QFileDialog.getOpenFileName(
            self, "Select Second Audio File", "", "Audio Files (*.wav *.flac *.ogg)"
        )
        if not self.file2:
            return None, None
        self.load_mix_song_2_button.setText(
            f"Load Song 2 for Mixing, Loaded File: {os.path.basename(self.file2)}"
        )
        self.check_mix_files_loaded()
        return self.file2

    def mix_audio(self):
        """
        Mix the two loaded audio files.a
        Stores the mixed audio data and sample rate as attributes.
        """
        audio1, sr1 = sf.read(self.file1)
        audio2, sr2 = sf.read(self.file2)

        if sr1 != sr2:
            num_samples = round(len(audio2) * float(sr1) / sr2)
            audio2 = resample(audio2, num_samples)
            sr2 = sr1
        max_length = max(len(audio1), len(audio2))
        audio1 = np.pad(
            audio1, ((0, max_length - len(audio1)), (0, 0)), mode="constant"
        )
        audio2 = np.pad(
            audio2, ((0, max_length - len(audio2)), (0, 0)), mode="constant"
        )
        weight1 = self.slider1.value() / 100.0
        weight2 = self.slider2.value() / 100.0
        mixed_audio = weight1 * audio1 + weight2 * audio2
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

        # Store the mixed audio data and sample rate as attributes
        self.mixed_audio = mixed_audio
        self.mixed_audio_sr = sr1

        # Process and compare the mixed audio
        self.process_and_compare_mixed_audio()

    def process_and_compare_mixed_audio(self):
        """
        Process and compare the mixed audio with the database.
        """
        # Create a temporary Song object with the mixed audio data
        self.song = Song("./Data/Temp/temp_mixed_audio.wav")
        self.song.file_path = "./Data/Temp/temp_mixed_audio.wav"
        self.song.duration = len(self.mixed_audio) / self.mixed_audio_sr

        # Save the mixed audio to a temporary file
        sf.write(self.song.file_path, self.mixed_audio, self.mixed_audio_sr)

        # Display the first letter of the mixed audio file name
        self.display_cover_image()

        self.compare_with_database()

    def play_audio(self):
        """
        Play the mixed audio.
        """
        print("Playing mixed audio...")
        mixed_audio, samplerate = sf.read(self.song.file_path)
        sd.play(mixed_audio, samplerate)
        self.stop_button.setEnabled(True)

    def stop_audio(self):
        """
        Stop the audio playback.
        """
        print("Stopping audio...")
        sd.stop()
        self.stop_button.setEnabled(False)

def process_all_songs_in_directory(directory):
    """
    Process all songs in the directory
    args:
        directory: str, directory containing the songs
    """
    for file_name in os.listdir(directory):
        if file_name.endswith((".mp3", ".wav", ".flac")):
            file_path = os.path.join(directory, file_name)
            logger.info(f"Processing {file_path}")

            song = Song(file_path)
            y, sr = song.load_audio()
            S_dB = song.generate_spectrogram(y, sr)
            song.save_spectrogram(S_dB, sr)
            mfccs_list, mel_spec_list = song.extract_features(S_dB, y, sr)
            song.save_features(mfccs_list, mel_spec_list)
            spectrogram_hash, mfcc_hash, mel_spec_hash = song.hash_features(
                mfccs_list, mel_spec_list
            )
            song.save_hashes(spectrogram_hash, mfcc_hash, mel_spec_hash)
            logger.info(f"Processed {file_path}")


def main():
    app = QApplication(sys.argv)
    with open("./Styles/index.qss", "r") as file:
        app.setStyleSheet(file.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Uncomment the following line to process all songs in the directory
    # process_all_songs_in_directory("./Data/Songs")
    main()
