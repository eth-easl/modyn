import pickle

import numpy as np
import os
PATH = "/scratch/fdeaglio/offline_dataset"

def main():
    mask = np.zeros((37189, 84))
    for file in os.listdir(PATH):
        if file.startswith("."):
            continue
        trigger = int(file.split("_")[1])
        indexes_and_weights = [el  for el in np.load(os.path.join(PATH, file), allow_pickle=False, fix_imports=False, mmap_mode="r")]
        for index, weight in indexes_and_weights:
            mask[index,trigger] = weight

    with open(f"/scratch/fdeaglio/selection_results/craig_mask.pkl", "wb") as f:
        pickle.dump(mask.tolist(), f)


if __name__ == "__main__":
    main()