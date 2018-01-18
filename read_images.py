# Deep learning lab course final project.
# Kaggle whale classification.

# routines to handle the images


import os
import pickle
import numpy as np
from scipy.misc import imread as depricated_imread  # works, but derpicated, see documentation
import imageio  # see https://imageio.github.io/ and http://imageio.readthedocs.io/en/latest/userapi.html
import skimage.color  # see http://scikit-image.org/docs/stable/api/api.html

#
#
#   FUNCTIONS
#
#


def load_jpg(path):
    """
    Loads a jpg to a numpy array.
    The shape of the numpy array is
    (x, y, 3) for RGB images and
    (x, y)    for grayvalue images.
    :param path: the file-path of the jpg
    :return: the numpy array representing the jpg
    """
    image = imageio.imread(path)
    # print(image.meta)
    # print(image.shape)
    return image


def test_imageio():
    """
    simple test of imageio. loads and writes files.
    """
    image = load_jpg("./data/small_train/0a5c0f48.jpg")
    imageio.imwrite('ch0.jpg', image[:, :, 0])
    imageio.imwrite('ch1.jpg', image[:, :, 1])
    imageio.imwrite('ch2.jpg', image[:, :, 2])
    imageio.imwrite('chA.jpg', image)


def test_skimage():
    """
    simple test of skimage. loads and writes files.
    """
    image = load_jpg("./data/small_train/0a5c0f48.jpg")
    # convert RGB to grayscale
    greyimage = skimage.color.rgb2gray(image)
    imageio.imwrite("greyimage.jpg", greyimage)


def pickle_images(folder="data/train", outfile="data/train.pkl"):
    """
    Don't use this, the uncompressed files get too large!
    Read jpeg images from folder and save in python format to
    outfile. The format used is a pickled numpy array containing the
    individual images as numpy arrays.
    """
    images = []
    files = os.listdir(folder)
    num_files = len(files)
    for i, fn in enumerate(files):
        file = os.path.join(folder, fn)
        print("Decompressing file {} of {}: {}".format(i, num_files, file))
        if os.path.isfile(file):
            images.append(depricated_imread(file))
    images = np.array(images)
    print("Saving in numpy/pickle format.")
    with open(outfile, "wb") as output:
        pickle.dump(images, output)
    print("Done reformatting images.")


def pickle_load_images(path="data/train.pkl"):
    with open(path, "rb") as infile:
        return pickle.load(infile)


#
#
#   GLOBAL CODE
#
#

# test_imageio()
# test_skimage()
