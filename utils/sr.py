import os
import librosa

def get_sample_rate(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return sr

def find_sample_rates(directory):
    sample_rates = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(root, file)
                sample_rate = get_sample_rate(file_path)
                sample_rates[file_path] = sample_rate
    return sample_rates

def rename_files_with_sample_rate(directory, sample_rates):
    for file_path, sample_rate in sample_rates.items():
        directory, file_name = os.path.split(file_path)
        new_file_name = f"{sample_rate}_{file_name}"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(file_path, new_file_path)

if __name__ == "__main__":
    directory = "./Data/Songs"
    sample_rates = find_sample_rates(directory)
    rename_files_with_sample_rate(directory, sample_rates)
    sorted_sample_rates = sorted(sample_rates.items(), key=lambda item: item[1])
    for file_path, sample_rate in sorted_sample_rates:
        print(f"{file_path}: {sample_rate} Hz")