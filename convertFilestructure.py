import os
import os.path
import shutil

BASE_PATH = "./data/"
TRAIN_DATA_PATH = "./data/train/"
_TRAIN_IMAGE_INDEX_FILE = "./data/train.csv" 


def copy_and_restructure(new_folder_path, limit_number_files=None):
    """
    Copy the whale images into a new folder with one subfolder for each whale.
    """
    with open(_TRAIN_IMAGE_INDEX_FILE, 'r') as file:
        indexfile = file.readlines()
    labels = dict()
    if os.path.exists(new_folder_path):
        raise EnvironmentError((1, "folder already exists:" + new_folder_path))
    os.mkdir(new_folder_path)
    for i in range(1, len(indexfile)):
        if len(indexfile) > 300 and i % 100 == 0:
            print("progress copying:", i, "of", len(indexfile))
        if limit_number_files is not None and i > limit_number_files:
            return
        indexline = indexfile[i]
        indexline = indexline.split(",")
        imagefilename = indexline[0]
        imagelabel = indexline[1]
        imagelabel = imagelabel.replace("\n", "")
        if imagelabel not in labels:
            os.mkdir(new_folder_path + imagelabel)
            labels[imagelabel] = True  # (imagelabel in labels)==True now
        shutil.copyfile(TRAIN_DATA_PATH + imagefilename, new_folder_path + imagelabel + "/" + imagefilename)
            


copy_and_restructure(BASE_PATH + "train_structured/", None)
