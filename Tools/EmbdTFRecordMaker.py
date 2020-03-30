import os 
import tensorflow as tf
import sys
import numpy as np

def generate_embd_tfrecords(data_fileneme, tfrecod_filename):
    fin = open(data_fileneme, "r")
    line = fin.readline()
    with tf.io.TFRecordWriter(tfrecod_filename) as f:
        embedding = []
        while line:
            tokens = line.strip().split(' ')
            for i in range(len(tokens)):
                embedding.append(float(tokens[i]))
            line = fin.readline()
        feature = {
            'raw_data': tf.train.Feature(float_list=tf.train.FloatList(value=embedding)),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        f.write(example.SerializeToString())
    f.close()


def _parse_embd_function(example_string):
    feature_description = {
        'raw_data': tf.io.FixedLenFeature([1551856*50], tf.float32),
    }

    feature_dict = tf.io.parse_single_sequence_example(example_string, feature_description)
   
    raw_data = feature_dict[0]['raw_data']
    return raw_data


def read_embd_tfrecords(tfrecod_filename, embed_shape):
    
    raw_dataset = tf.data.TFRecordDataset(tfrecod_filename)
   
    parsed_dataset = raw_dataset.map(_parse_embd_function)

    for x in parsed_dataset:
         _data = x
         break

    return tf.reshape(_data, embed_shape).numpy()

        
if __name__ == "__main__":
    generate_embd_tfrecords(sys.argv[1], sys.argv[2])
    #print(read_embd_tfrecords(sys.argv[2], [1551856,50]))