import sys
import soundfile as sf
import numpy as np
from scipy.signal import resample
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget


class AudioMixer(QWidget):
    def __init__(self):
        super().__init__()




def main():
    app = QApplication(sys.argv)
    mixer = AudioMixer()
    file1, file2 = mixer.load_files()
    if file1 and file2:
        mixer.mix_audio(file1, file2)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
