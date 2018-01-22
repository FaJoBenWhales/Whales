# Deep learning lab course final project.
# Kaggle whale classification.
#
#
# Interface to load wale data for deep learning. The data will be preprocessed.
#
#


def load_train_batch(batch_size, processingOptions):
    """
    Loads a batch of wale data. Applies data augmentation and pre-processing.
    :param batch_size: The number of images. Image will be numpy array of shape (x, y) or (x, y, 3).
    :param processingOptions: dictionary of type (String -> optionArgument).
    optionArgument depends on the option. Options may be combined without limitation.
    Options are for now:
    "augmentation": factor. factor gives how many images to generate per given image.
    "rescale": [x, y]. x and y gives the desired image dimensions. Scales the images.
    "autocrop" [x, y]. DON'T EXPECT IMPLEMENTATION SOON. x and y gives the desired image dimensions. Crops the image to wale flukes.
    "greyimage": bool. If bool==True always returns a grey image. Otherwise a color or grey image.
    :return: A numpy array of shape (batch_size, x, y) or (batch_size, x, y, 3).
    """
    raise NotImplementedError  # TODO implement
