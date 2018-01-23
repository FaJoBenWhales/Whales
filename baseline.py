# Deep learning lab course final project.
# Kaggle whale classification.

# Use pretrained tensorflow model (Inception v3) and just learn the
# last (= classifictaion) layers as described in https://www.tensorflow.org/tutorials/image_retraining

# Images are expected in a directory structure sorted by label
# (e.g. whale1/image*.jpg, whale2/iamge*.jpg, ...)

from utilities import get_whales, read_csv
from operator import itemgetter
import shutil
import os
import subprocess
import sys

def sort_by_labels(num_labels,
                   include_new_whale=False,
                   labels_path="data/train.csv",
                   image_path="data/train",
                   directory="baseline"):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    train_list = read_csv(labels_path)
    whales, _ = get_whales(train_list)
    whales = sorted(whales, key=itemgetter(1), reverse=True)
    print("Creating directories for {} most common whales.".format(num_labels))
    for name, _, image_ids in whales[1:num_labels+1]:
        path = os.path.join(directory, name)
        print("Creating directory {}: {} images.".format(name, len(image_ids)))
        if not os.path.isdir(path):
            os.mkdir(path)
        for ids in image_ids:
            filename = train_list[ids][0]
            shutil.copyfile(os.path.join(image_path, filename), os.path.join(path, filename))
    


def train_classifier():
    cmd = ["python3", "retrain.py", "--image_dir", "baseline", "--validation_batch_size=-1", "--how_many_training_steps=8000"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        

sort_by_labels(300, include_new_whale=True)
print("Done creating environment.")
#print("Retrain last layer of Inception v3 model.")
#train_classifier()
