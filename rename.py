#!/usr/bin/env python3
import os


def main():
    base_dir = "/scratch/sjohn/modyn/datasets"
    # List of ending patterns to check
    endings = ("10", "20", "30", "40", "50")

    # Iterate over subdirectories in base_dir that start with "dmixreview"
    for folder in os.listdir(base_dir):
        if folder.startswith("dmixreview"):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder_path}")
                for filename in os.listdir(folder_path):
                    filepath = os.path.join(folder_path, filename)
                    if os.path.isfile(filepath):
                        base, ext = os.path.splitext(filename)
                        # Check if the basename ends with one of the endings and doesn't already have a .csv extension
                        if ext.lower() != ".csv":
                            new_filename = base + ".csv"
                            new_filepath = os.path.join(folder_path, new_filename)
                            os.rename(filepath, new_filepath)
                            print(f"Renamed {filename} -> {new_filename}")


if __name__ == "__main__":
    main()
