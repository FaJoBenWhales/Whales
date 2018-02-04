# Deep learning lab course final project.
# Kaggle whale classification.

import numpy as np
import os.path
from scipy.misc import imresize
from keras.preprocessing.image import array_to_img
from keras import backend as K

def random_crop_generator(image_generator, target_size=(299, 299),
                          max_crop_x=0.5, max_crop_y=0.5,
                          interpolation_method='nearest',
                          save_to_dir=None,
                          preserve_aspect_ratio=False):
    """Apply random crops to image ndarrays from given generator.
    image_generator: generator yielding numpy.ndarray images 
                     (e.g. Keras ImageDataGenerator.flow_from_directory)
    target_size: desired output size
    max_crop_x, max_crop_y: maximum percentage of cropped away (lost) pixels in
                            x / y dimension
    interpolation_method: interpolation method used by scipy.misc.imresize

    Returns: numpy ndarray image (as a generator)."""

    if preserve_aspect_ratio:
        raise NotImplementedError(
            "Preserving aspect ratio is currently not available.")
    
    for batch_images, batch_labels in image_generator:
        length = batch_images.shape[0]
        new_images = np.zeros(tuple([length] +
                                    list(target_size) +
                                    list(batch_images.shape)[3:]),
                              dtype=K.floatx())
                
        for i in range(length):
            image = batch_images[i]
            
            # calculate crop indices
            size_x, size_y = image.shape[:2]
            lose_x = np.random.randint(size_x * max_crop_x + 1)
            lose_y = np.random.randint(size_y * max_crop_y + 1)
            start_x = np.random.randint(lose_x + 1)
            start_y = np.random.randint(lose_y + 1)
            end_x = size_x - lose_x + start_x
            end_y = size_y - lose_y + start_y
            
            # apply the crop
            image = image[start_x:end_x, start_y:end_y, :]
            
            if save_to_dir:
                img = array_to_img(image)
                fname = '{hash}.png'.format(hash=np.random.randint(1e7))
                img.save(os.path.join(save_to_dir, fname))
                
            image = imresize(image, size=target_size,
                             interp=interpolation_method)
            new_images[i] = image

        yield new_images, batch_labels
