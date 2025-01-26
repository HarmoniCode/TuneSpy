import os
import zipfile

def compress_directory(directory, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=directory)
                zipf.write(file_path, arcname)
                print(f"Compressed {file_path} as {arcname}")

if __name__ == "__main__":
    directory_to_compress = "./Data"
    output_zip_file = "data_archive.zip" 
    compress_directory(directory_to_compress, output_zip_file)
    print(f"All files in {directory_to_compress} have been compressed into {output_zip_file}")