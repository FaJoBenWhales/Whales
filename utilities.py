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
def plot_whales(imgs, max_imgs = 100, labels=None, rows=4, figsize=(16,10)):
    
    figure = plt.figure(figsize=figsize)
    imgs = imgs[:max_imgs]
    cols = len(imgs) // rows + 1
    # print("gaga")
    for i in range(len(imgs)):
    # for i in range(10)
    # for i, img in enumerate(imgs):    
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        plt.imshow(imgs[i], cmap='gray')              
        if labels != None and len(labels)>0:
            subplot.set_title(labels[i], fontsize=16)
    plt.show()
            
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
# creating subdirectory for each class/whale, as required by Keras/Tensorflow
def create_small_case(sel_whales = [1,2,3],          # whales to be considered
                      train_dir = "data/train",
                      train_csv = "data/train.csv",
                      small_dir = "data/small_train", 
                      small_csv = None,         # optionally write CSV File as train.csv
                      sub_dirs = True):         # create subdirectory for each class

    if not os.path.isdir(train_dir):
        print("{} no valid directory".format(train_dir))
        return

    try: 
        shutil.rmtree(small_dir)   # remove directory, if already existing 
        print("old directory removed {}".format(small_dir))
    except:
        print("directory {} did not exist so far".format(small_dir))
        
    os.mkdir(small_dir)
    if not sub_dirs:
        target_dir = os.path.join(small_dir, "whales")
        os.mkdir(target_dir)
        
    small_list=[]
    train_list = read_csv(file_name = train_csv)  # get list with (filenames, whalenames)
    whales, counts = get_whales(train_list)   # get list of whales ordered by frequency
    for i in sel_whales:                          
        print("copy {} images for whale # {} in ordered list, called {}"
              .format(whales[i][1], i, whales[i][0]))
        if sub_dirs:
            target_dir = os.path.join(small_dir, whales[i][0])
            os.mkdir(target_dir)
        for idx in whales[i][2]:   # array of indices of this whale pointing into train_csv list
            fn = train_list[idx][0]     # get filename out of train_csv list
            shutil.copy(os.path.join(train_dir, fn), 
                        os.path.join(target_dir, fn))
            
            small_list.append((fn,whales[i][0]))
    if small_csv != None:
        print("write csv file: {}".format(small_csv))            
        write_csv(small_list, small_csv)
    
    print(len(small_list), " images of ", len(sel_whales), " whales copied")
    return len(small_list)  # return number of images


# create dictionary mapping class_no as used by Keras in predictions --> whalenames
def make_label_dict(directory="data/model_train"):
    label_dict = dict()
    for i, label in enumerate(sorted(os.listdir(directory))):
        label_dict[i] = label
    return label_dict



'''
evaluation metrics MAP@5
sources: 
https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
https://www.kaggle.com/c/FacebookRecruiting/discussion/2002
https://en.wikipedia.org/wiki/Information_retrieval
Note, that the metric is designed for "document retrieval", 
where many outcomes might be true (= "relevant documents")
Our case is specific, as there is only one "relevant document" per prediction 
(= the true prediction)
'''
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

'''
As a benchmark create case of a dumb model without any predictive power and test, how this model performs
measured in MAP@5 metric, by building 
1) a dummy set of true labels and 
2) a dummy model prediction (max pred ranked predictions per sample) 
and applying mean_average_precision() on these


compute:
1) a "dumb" model, that maps the images to the individuals uniform randomly, 
2) a model, that maps the images to the individuals along a probability distribution 
   that reflects the distribution of frequencies of observed whales as given in "train.csv"
   --> whale #1, will occure 34 times more likely than whale #2.500 with one image
3) like 2), but without the category "new whale" (~800 occurences) 
'''
from scipy.stats import rv_discrete

def Dummy_MAP(probs = 'uniform', 
              distributed_as = "data/train.csv",    # csv file with mimiked distribution
              max_pred = 5,
              image_no = 100):
    
    dist_list = read_csv(file_name = distributed_as)   # for testing whole train data set

    whales, counts = get_whales(dist_list)
    if probs == 'weighted_without_first':   # leaving out first whale "new whale" with 810 individuals
        whales = whales[1:]
        counts = counts[1:]
    total_whales = np.sum(counts)
    sorted_whales = np.arange(0,len(whales))     

    # create dummy list of true labels following probability distribution
    dummy_true_labels = []
    # probability distribution = normalised distribution of frequencies in dist_list
    px=[whales[i][1]/total_whales for i in range(len(whales))]
    for i in range(image_no):
        # sample labels according probability distribution defined above
        dummy_true_labels.append(rv_discrete(values=(sorted_whales,px)).rvs(size=1)[0])
    
    # to each image in dummy predictions map a ranked list of max_pred whales
    # as random number between 1 and # of individuals in scenario (indeces in whale list)
    # following the distribution as given in "distributed_as"
    dummy_preds = []
    if probs == 'uniform':
        for i in range(image_no):
            dummy_preds.append(np.random.randint(0,len(counts),max_pred))

    # mimik prob. distribution from distribution in train.csv
    elif probs == 'weighted' or probs == 'weighted_without_first':  
        # probability distribution = normalised distribution of frequencies
        px=[whales[i][1]/total_whales for i in range(len(whales))]
        # create ranked list of max_pred = 5 labels per image in train_list
        for i in range(image_no):
            # sample labels according to probability distribution defined above
            ranks=rv_discrete(values=(sorted_whales,px)).rvs(size=max_pred)  
            dummy_preds.append(ranks)
            
    else:
        raise AssertionError

    return mean_average_precision(dummy_preds, dummy_true_labels, max_pred)