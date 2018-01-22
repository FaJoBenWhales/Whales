# Deep learning lab course final project.  Kaggle whale
# classification.

# utility functions

import os
import csv
import numpy as np
import shutil
import matplotlib.pyplot as plt

# read train.csv, return list of tuples (filename,whale_name)
def read_csv(file_name = "data/train.csv"):

    if not os.path.isfile(file_name):
        print("{} no valid path".format(file_name))
        return None
    
    csv_list = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            csv_list.append((rows[0],rows[1]))

    return csv_list[1:]

# write list of tuples (filename,whale_name) to csv file
def write_csv(csv_list, file_name = "data/small_train.csv"):
    
    if os.path.isfile(file_name):
        os.remove(file_name)
    
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Image'] + ['Id'])
        for entry in csv_list:
            writer.writerow([entry[0]] + [entry[1]])
            #spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
            #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


# generate sorted list clustered by individuals: 
# (name, number of images, array of indeces into train_list)
def get_whales(train_list):
    
    train_arr = np.asarray(train_list)
    whale_names = np.unique(train_arr.T[1], axis=0)

    whales = []
    for name in whale_names:
        idx = np.where(train_arr.T[1] == name)[0]
        whales.append((name,idx.shape[0],idx)) 

    # sort by frequency of occurence in descending order 
    whales.sort(key=lambda x:x[1], reverse=True)
    counts=[whale[1] for whale in whales]      # list of numbers of individuals ([34,25,24...])

    return whales, counts

def show_histogram(num = 100, file_name = "data/train.csv"):
    train_list = read_csv(file_name = file_name)
    _, counts = get_whales(train_list)    
    plt.hist(counts[1:num], bins=counts[1], color="b", align = "left", rwidth=0.75)  # skip first entry "new whale"
    plt.title("{} most frequent whales".format(num))
    plt.xlabel('number of images per individual')
    plt.ylabel('number of individuals')
    plt.show()

# alternative representation of frequencies of occurance    
def show_frequencies(num = 100, file_name = "data/train.csv"):
    train_list = read_csv(file_name = file_name)
    _, counts = get_whales(train_list)    
    num = min(num, len(counts)-1)   # avoid errors if num chosen larger than len(counts)
    plt.bar(np.arange(num),counts[1:num+1], color = 'b', edgecolor = 'b')
    plt.title("number of images per whale".format(num))
    plt.xlabel('individuals')
    plt.ylabel('number of images per individual')
    plt.show()

# plot list of given images    
def plot_whales(imgs, labels=None, rows=4):
    
    figure = plt.figure(figsize=(16, 10))
    cols = len(imgs) // rows + 1
    
    for i, img in enumerate(imgs):    
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        plt.imshow(img, cmap='gray')              
        if len(labels)>0:
            subplot.set_title(labels[i], fontsize=16)
            
# plot (all or first max_imgs) images of whale with number whale_no
def show_whales(whale_no, folder="data/train", csv_file="data/train.csv",
                max_imgs=100, rows=4, labels=False):
    
    train_list = read_csv(file_name = csv_file)
    whales, counts = get_whales(train_list)   # get list of whales ordered by frequency    
    print("Images of whale {}, number {} in list".format(whales[whale_no][0], whale_no))

    img_count = min(counts[whale_no], max_imgs)

    imgs = []
    label_list = []      # for future extensions with more than one individual
    idx = whales[whale_no][2]  
    for i in range(img_count):        
        fn = train_list[idx[i]][0]     # get filename out of train_csv list
        path = os.path.join(folder, fn)
        if os.path.isfile(path):
            imgs.append(plt.imread(path))
            if labels: 
                label_list.append(whales[whale_no][0])     
        else:
            print("invalid path: {}".format(path))
    
    plot_whales(imgs, labels=label_list, rows=rows)

           
# as a playground reproduce setting (image-files, csv-file, directory structure) 
# for small case (small number of selected individuals)
def create_small_case(sel_whales = [1,2,3],          # whales to be considered
                      train_dir = "data/train",
                      train_csv = "data/train.csv",
                      small_dir = "data/small_train", 
                      small_csv = "data/small_train.csv"):

    if not os.path.isdir(train_dir):
        print("{} no valid directory".format(train_dir))
        return

    try: 
        shutil.rmtree(small_dir)   # remove directory, if already existing 
        print("old directory removed {}".format(small_dir))
    except:
        print("directory {} did not exist so far".format(small_dir))
    
    os.mkdir(small_dir)
    small_list=[]
    train_list = read_csv(file_name = train_csv)  # get list with (filenames, whalenames)
    whales, counts = get_whales(train_list)   # get list of whales ordered by frequency
    for i in sel_whales:                          
        # whale_idx = whales[i][2]                 # get list of indices of this whale
        print("copy {} images for whale # {} in ordered list, called {}"
              .format(whales[i][1], i, whales[i][0]))
        for idx in whales[i][2]:        # np array containing indices of this whale pointing into train_csv list  
            fn = train_list[idx][0]     # get filename out of train_csv list
            shutil.copy(os.path.join(train_dir, fn), 
                        os.path.join(small_dir, fn))
            
            small_list.append((fn,whales[i][0]))
    print("write csv file: {}".format(small_csv))            
    write_csv(small_list, small_csv)

# evaluation metrics    
# Precision at k is a percentage of correct items among first k recommendations
# "cut off k": only first occurence of matching prediction contributes to score
def precision_at_k(model_prediction, true_label, k):
    if model_prediction[k] == true_label and not true_label in model_prediction[:k]:
        return(1/(k+1))   # return precision = TP / total number of samples
    else:
        return(0)

# precision at cut-off k 
def average_precision(model_prediction, true_label, max_pred):
    average_precision = 0
    for k in range(max_pred):
        average_precision += precision_at_k(model_prediction, true_label, k)
    return average_precision # / min(len(true_label), max_pred)

# mean average precision is simply the mean of average_precision over several samples
def mean_average_precision(model_predictions, true_labels, max_pred):
    return np.mean([average_precision(mp, tl, max_pred) 
                    for mp, tl in zip(model_predictions, true_labels)])

