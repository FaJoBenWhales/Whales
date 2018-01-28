# Deep learning lab course final project.
# Kaggle whale classification.
#
#
# Interface to load whale data for deep learning. The data will be preprocessed.
#
#

import skimage.transform
import skimage.color
import skimage.io
# import imageio


# constants
_TRAIN_IMAGE_INDEX_FILE = "./data/train.csv"
_TRAIN_IMAGE_FOLDER = "./data/train/"

# module global variables
_traindata_offset = 0  # where to start reading next batch of train data


def _resize(image, x_res, y_res):
    """
    Adapt the resolution of an image.
    :param image: The image to scale down. Numpy array with dim (x, y) or (x, y, 3).
    :param x_res: The resulting number of pixels in x direction.
    :param y_res: The resulting number of pixels in y direction.
    :return: The image with reduced resolution.
    """
    return skimage.transform.resize(image, (x_res, y_res), mode='edge', clip=True, preserve_range=True)


def _get_train_images(batch_size: int) -> list:
    """
    Load train images and labels.
    :param batch_size: number of images to load.
    :return:  A list of tuples. The tuples are (image, label). image is a numpy array of dimension (x, y) for 
        greyimages and (x, y, 3) for color images. label is a string. 
    """
    global _traindata_offset
    with open(_TRAIN_IMAGE_INDEX_FILE, 'r') as file:
        indexfile = file.readlines()
    number_train_images = len(indexfile) - 1  # there is one header line
    batch = []
    for _ in range(batch_size):
        # read images
        indexline = indexfile[_traindata_offset + 1]
        _traindata_offset += 1
        if _traindata_offset > number_train_images:
            _traindata_offset = 0
        indexline = indexline.split(",")
        imagefilename = indexline[0]
        imagelabel = indexline[1]
        image = skimage.io.imread(_TRAIN_IMAGE_FOLDER + imagefilename)
        batch.append((image, imagelabel))
    return batch


def load_train_batch(batch_size: int, processing_options: dict) -> list:
    """
    Loads a batch of whale data. Applies data augmentation and pre-processing.
    Successive calls to the function iterate over the data.
    :param batch_size: The number of images.
    :param processing_options: dictionary of type (String -> optionArgument).
        optionArgument depends on the option. Options may be combined without limitation.
        Options are for now:
        "augmentation": factor. factor gives how many images to generate per given image.
        "rescale": [x, y]. x and y gives the desired image dimensions. Scales the images.
        "autocrop" [x, y]. DON'T EXPECT IMPLEMENTATION SOON. x and y gives the desired image dimensions. Crops the image to wale flukes.
        "greyimage": bool. If bool==True always returns a grey image. Otherwise a color or grey image.
    :return: A list of tuples. The tuples are (image, label). image is a numpy array of dimension (x, y) for 
        greyimages and (x, y, 3) for color images. label is a string. 
    """

    images = _get_train_images(batch_size)

    # processing pipeline
    if "augmentation" in processing_options:
        raise NotImplementedError
    if "rescale" in processing_options:
        images_rescaled = []
        rescale_options = processing_options["rescale"]
        x = rescale_options[0]
        y = rescale_options[1]
        for image in images:
            images_rescaled.append((_resize(image[0], x, y), image[1]))
        images = images_rescaled
    if "autocrop" in processing_options:
        raise NotImplementedError
    if "greyimage" in processing_options:
        images_grey = []
        convert = processing_options["greyimage"]
        if convert:
            for image in images:
                images_grey.append((skimage.color.rgb2gray(image[0]), image[1]))
            images = images_grey

    return images


# noinspection PyBroadException
def module_test():
    """
    Tests this module.
    Prints messages about found issues to stdout. Keeps quiet else.
    """ 
    general_message = "In module Dataloader.py a test-case failed: "    
    
    # testcase 1
    try:
        images = load_train_batch(100, dict())
        if len(images) != 100:
            raise AssertionError
    except Exception:
        print(general_message + "case number 1")
    
    # testcase 2
    try:
        images = load_train_batch(50, {"rescale": [230, 230]})
        for image in images:
            if image[0].shape != (230, 230) and image[0].shape != (230, 230, 3):
                raise AssertionError
    except Exception:
        print(general_message + "case number 2")
        
    # testcase 3
    try:
        images = load_train_batch(70, {"rescale": [230, 230], "greyimage": True})
        for image in images:
            if image[0].shape != (230, 230):
                raise AssertionError
    except Exception:
        print(general_message + "case number 3")


if __name__ == "__main__":
    module_test()
