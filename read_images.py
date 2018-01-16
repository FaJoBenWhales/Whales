# Deep learning lab course final project.
# Kaggle whale classification.

# Read jpeg images into numpy arrays.

# We cannot use this like this, the uncompressed files get too large!

import os
import pickle
import numpy as np
from scipy.misc import imread


def pickle_images(folder="data/train", outfile="data/train.pkl"):
    """Read jpeg images from folder and save in python format to
outfile. The format used is a pickled numpy array containing the
individual images as numpy arrays. """
    images = []
    files = os.listdir(folder)
    num_files = len(files)
    for i, fn in enumerate(files):
        file = os.path.join(folder, fn)
        print("Decompressing file {} of {}: {}".format(i, num_files, file))
        if os.path.isfile(file):
            images.append(imread(file))
    images = np.array(images)
    print("Saving in numpy/pickle format.")
    with open(outfile, "wb") as output:
        pickle.dump(images, output)
    print("Done reformatting images.")
    
    
def load_images(path="data/train.pkl"):
    with open("train.pkl", "rb") as infile: 
        return pickle.load(infile)

# pickle_images("data/train")
pickle_images("data/small_train")
print(load_images())
