# Deep learning lab course final project.
# Kaggle whale classification.
#
#
# Interface to load whale data for deep learning. The data will be preprocessed.
#
#

import skimage.transform
import skimage.color
import imageio


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
    Load train images.
    :param batch_size: number of images to load.
    :return: list of ndarrays. len(list) = batch_size, ndarrays have image-shapes.
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
        image = imageio.imread(_TRAIN_IMAGE_FOLDER + imagefilename)
        batch.append(image)
    return batch


def load_train_batch(batch_size: int, processing_options: dict) -> list:
    """
    Loads a batch of whale data. Applies data augmentation and pre-processing.
    Successive calls to the function iterate over the data.
    :param batch_size: The number of images. Image will be numpy array of shape (x, y) or (x, y, 3).
    :param processing_options: dictionary of type (String -> optionArgument).
    optionArgument depends on the option. Options may be combined without limitation.
    Options are for now:
    "augmentation": factor. factor gives how many images to generate per given image.
    "rescale": [x, y]. x and y gives the desired image dimensions. Scales the images.
    "autocrop" [x, y]. DON'T EXPECT IMPLEMENTATION SOON. x and y gives the desired image dimensions. Crops the image to wale flukes.
    "greyimage": bool. If bool==True always returns a grey image. Otherwise a color or grey image.
    :return: A list of numpy arrays. Combined shape (batch_size, x, y) or (batch_size, x, y, 3).
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
            images_rescaled.append(_resize(image, x, y))
        images = images_rescaled
    if "autocrop" in processing_options:
        raise NotImplementedError
    if "greyimage" in processing_options:
        images_grey = []
        convert = processing_options["greyimage"]
        if convert:
            for image in images:
                images_grey.append(skimage.color.rgb2gray(image))
            images = images_grey

    return images
