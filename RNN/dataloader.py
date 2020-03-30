import sys
import numpy as np

from TFRecordMaker import read_tfrecords

def load_embeddings(filename):
    return np.loadtxt(filename, delimiter = ' ')

def load_dataset(tfrecordfile):
    train_dataset = read_tfrecords(tfrecordfile)
    return train_dataset
