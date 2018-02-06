from keras.applications.inception_v3 import InceptionV3
import sys

RANGE = 20


def load_model():
    pretrained_model = InceptionV3(weights='imagenet', include_top=False)
    return


def simple_test():
    for i in range(RANGE):
    print("load model for the {}. time".format(i+1))
    load_model()



if __name__ == "__main__":
    if "--simple" in sys.argv:
        simple_test()
