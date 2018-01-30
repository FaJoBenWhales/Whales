# Deep learning lab course final project.
# Kaggle whale classification.

import os
import sys
import numpy as np

#import h5py
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

import utilities as ut
import keras_tools as tools

# Use pretrained model as described in https://keras.io/applications/

def create_pretrained_model(two_layers=True, num_classes=10):
    # load pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    if two_layers:
        # recommended by
        # https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # train only the top classifier layers
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to
    # non-trainable)
    # metrics='accuracy' causes the model to store and report accuracy (train
    # and validate)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cross_validation(model, epochs=20, cross_validation_iterations=5,
                           num_classes=10, save_path=None):
    histories = []
    for i in range(cross_validation_iterations):
        path_components = save_path.split(os.extsep)
        path = ".".join(path_components[:-1] + [i] + path_components[-1])
        hist = train_and_save(model, epochs, num_classes, save_path)
        histories.append(hist)
    return histories


def train(model, epochs=20, num_classes=10, save_path=None, batch_size=16,
          train_dir="data/model_train", valid_dir="data/model_valid"):
    # create new environment with new random train / valid split
    num_train_imgs, num_valid_imgs = tools.prepare_environment(num_classes)

    train_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,   # redundant with featurewise_center ? 
        # preprocessing_function=preprocess_input, not used in most examples
        # horizontal_flip = True,    # no, as individual shapes are looked for
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        # save_to_dir="data/model_train/augmented",    
        # color_mode="grayscale",
        target_size=(299,299),
        batch_size=batch_size, 
        class_mode="categorical")

    valid_gen = image.ImageDataGenerator(
        rescale=1./255,
        fill_mode="nearest")

    valid_flow = valid_gen.flow_from_directory(
        valid_dir,
        target_size=(299,299),
        class_mode="categorical") 

    hist = model.fit_generator(
        train_flow, 
        steps_per_epoch=num_train_imgs//batch_size,
        verbose=2, 
        validation_data=valid_flow,   # to be used later
        validation_steps=num_valid_imgs//batch_size,
        epochs=epochs)

    if save_path is not None:
        model.save(save_path)
        
    return hist.history


def main():
    print("Run complete script: Printing data info, train, print test results.")
    tools.print_data_info()
    model = create_pretrained_model()
    histories = train(model, epochs=2)
    print(histories)
    run_name = tools.get_run_name()
    tools.save_learning_curves(histories, run_name)
    ut.write_csv_dict(histories,
                      keys=['loss', 'acc', 'val_loss', 'val_acc'],
                      filename=run_name + '.csv')
    tools.print_model_test_info(model)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
        exit()
    if "--prepare-only" in sys.argv:
        tools.prepare_environment()
        exit()
    if "--class-graph" in sys.argv:
        tools.draw_num_classes_graphs()
        exit()
    print("given command line options unknown.")
    
