import os 
import tensorflow as tf
import sys
import numpy as np

def generate_tfrecords(data_fileneme, tfrecod_filename):
    fin = open(data_fileneme, "r")
    line = fin.readline()
    with tf.io.TFRecordWriter(tfrecod_filename) as f:
        while line:
            tokens = line.strip().split(' ')
            label = int(tokens[0])
            input_x = tokens[1:]
            for i in range(len(input_x)):
                input_x[i] = int(input_x[i]) 
            feature = {
                'raw_data': tf.train.Feature(int64_list=tf.train.Int64List(value=input_x)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            f.write(example.SerializeToString())
            line = fin.readline()
    f.close()
    fin.close()


def _parse_function(example_string):
    feature_description = {
        'raw_data': tf.io.FixedLenFeature([10], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    feature_dict = tf.io.parse_single_sequence_example(example_string, feature_description)
   
    raw_data = feature_dict[0]['raw_data']
    label = feature_dict[0]['label']
    return raw_data, label


def read_tfrecords(tfrecod_filename):
    
    raw_dataset = tf.data.TFRecordDataset(tfrecod_filename)
    # for raw_record in raw_dataset.take(5):
    #     print(repr(raw_record))

    parsed_dataset = raw_dataset.map(_parse_function)
    # for x,y in parsed_dataset:
    #      print(x)
    #      print(y) 
    
    # for parsed_record in parsed_dataset.take(5):
    #     print(repr(parsed_record))

    return parsed_dataset

        
if __name__ == "__main__":
    generate_tfrecords(sys.argv[1], sys.argv[2])
    read_tfrecords(sys.argv[2])