# Deep learning lab course final project.
# Kaggle whale classification.

# Helper functions for the main keras model.

import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras import backend as K

import utilities as ut

# global variables
train_dir = "data/model_train"
train_csv = "data/model_train.csv"
valid_dir = "data/model_valid"
valid_csv = "data/model_valid.csv"
max_preds = 5    # number of ranked predictions

# Create training environment for training data.
def prepare_environment(num_classes=10, data_split_ratio=0.8):
    """Create environment for keras data generator flow.
    data_split_ratio: percentage of files which become train (not
    validation) images.
    Returns number of train and validation images as (int, int).
    """
    return ut.create_small_case(
        sel_whales=np.arange(1, num_classes + 1),  # whales to be considered
        train_dir=train_dir,
        train_csv=train_csv,
        valid_dir=valid_dir,
        valid_csv=valid_csv,
        train_valid=data_split_ratio,
        sub_dirs=True)

def get_run_name(additional=""):
    if additional != "":
        additional = "_" + additional
    return "run-{}{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
        additional)


def save_learning_curves(history, run_name, base_path="plots/"):
    """Saves the data from keras history dict in loss and accuracy graphs to folder
    specified by base_path and run_name."""
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    losses = {k: history[k] for k in ['loss', 'val_loss']}
    accuracies = {k: history[k] for k in ['acc', 'val_acc']}
    x = range(len(losses['loss']))
    fn_losses = os.path.join(path, "loss.png")
    fn_accuracies = os.path.join(path, "accuracy.png")
    ut.save_plot(x, ys=losses, xlabel="epoch", ylabel="loss",
                 title=run_name, path=fn_losses)
    ut.save_plot(x, ys=accuracies, xlabel="epoch", ylabel="accuracy",
                 title=run_name, path=fn_accuracies)


def draw_num_classes_graphs():
    """Train network and save learning curves for different values for num_classes."""
    values = [10, 50, 100, 250, 1000, 4000]
    for num_classes in values:
        print("Training model on {} most common classes.".format(num_classes))
        model = create_pretrained_model(num_classes=num_classes)
        histories = train(model, epochs=50, num_classes=num_classes)
        run_name = get_run_name("{}classes".format(num_classes))
        save_learning_curves(histories, run_name)
        csv_path = os.path.join("plots/", run_name, "data.csv")
        ut.write_csv_dict(histories,
                       keys=['loss', 'acc', 'val_loss', 'val_acc'],
                       filename=csv_path)


def print_data_info(train_dir="data/model_train", batch_size=16):
    # define image generator
    train_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale = 1./255,   # redundant with featurewise_center ? 
        # preprocessing_function=preprocess_input, not used in most examples
        # horizontal_flip = True,    # no, as individual shapes are looked for
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)
    
    # train the model on the new data for a few epochs
    train_flow = train_gen.flow_from_directory(
        train_dir,
        # save_to_dir="data/model_train/augmented",    
        # color_mode="grayscale",
        target_size=(299,299),
        batch_size=batch_size, 
        class_mode="categorical")

    # valid_gen = image.ImageDataGenerator(
    #     rescale = 1./255,
    #     fill_mode = "nearest")

    # valid_flow = valid_gen.flow_from_directory(
    #     valid_dir,
    #     target_size = (299,299),
    #     class_mode = "categorical")


    # get dict mapping whalenames -> class_no:
    whale_class_map = (train_flow.class_indices)
    # get dict mapping class_no -> whalenames:
    class_whale_map = ut.make_label_dict(directory=train_dir)
    print("whale_class_map:")
    print(whale_class_map)
    print("class_whale_map:")
    print(class_whale_map)
    #print("num_train_imgs:")
    #print(num_train_imgs)


def print_model_test_info(model, num_classes=10, steps=10, batch_size=16):
    # try to verify on test data --> no success so far
    
    # use all training data of the first num_classes whales a test data.
    # no good practice, but all training data have been augmented, so at least some indication
    # about predictive power of model
    test_dir = "data/model_test"
    test_csv = "data/model_test.csv"
    num_train_imgs, num_valid_imgs = ut.create_small_case(
        sel_whales=np.arange(1, num_classes + 1),  # whales to be considered
        train_dir=test_dir,
        train_csv=test_csv,
        valid_dir=None,     # no validation, copy all data into test_dir "data/model_test"
        valid_csv=None,
        train_valid=1.,
        sub_dirs=True) 
    
    # for test Purposes !!!

    # valid_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,
        # preprocessing_function=preprocess_input,   # model specific function
        fill_mode="nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        # color_mode="grayscale",
        batch_size=batch_size,     
        target_size=(299,299),
        class_mode="categorical")    # use "None" ??
    
    preds = model.predict_generator(test_flow, verbose=1, steps=steps)

    whale_class_map = (test_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = ut.make_label_dict(directory=test_dir) # get dict mapping class_no --> whalenames
    print("whale_class_map:")
    print(whale_class_map)
    print("class_whale_map:")
    print(class_whale_map)
    print("preds.shape:")
    print(preds.shape)
    print("preds[:10]")
    print(preds[:10])
    
    # ge list of model predictions: one ordered list of maxpred whalenames per image
    top_k = preds.argsort()[:, -max_preds:][:, ::-1]
    # top_k = preds.argsort()[:, -max_preds:]
    model_preds = [([class_whale_map[i] for i in line]) for line in top_k]  

    # get list of true labels: one whalename per image
    test_list = ut.read_csv(file_name = test_csv)    # list with (filename, whalename)
    true_labels = []
    for fn in test_flow.filenames:
        offset, filename = fn.split('/')
        whale = [line[1] for line in test_list if line[0] == filename][0]
        true_labels.append(whale)

    print("model predictions: \n", np.array(model_preds)[0:20])
    print("true labels \n", np.array(true_labels)[0:20])

    MAP = ut.mean_average_precision(model_preds, true_labels, max_preds)
    print("MAP", MAP)

    for i in range(10):
        Dummy_map = ut.Dummy_MAP(probs = 'weighted', distributed_as = train_csv, image_no = len(test_list))
        print("Dummy MAP weighted", Dummy_map)

    # MAP only slightly higher than average dummy MAP
