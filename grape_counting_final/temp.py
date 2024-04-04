import os
import shutil


def move_files(source_dir, destination_dir):
    files = os.listdir(source_dir)
    for file in files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)


source_directory = "./datasets/test_dataset"
destination_directory = "./datasets"
move_files(source_directory, destination_directory)
