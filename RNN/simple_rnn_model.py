import tensorflow as tf
import numpy as np

class SimpleRNN(tf.keras.Model):
    def __init__(self, vocal_size, embd_size, units, embd_table, is_mask):
        super().__init__()
        #self.flatten = tf.keras.layers.Flatten()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocal_size,output_dim=embd_size, weights=[embd_table], mask_zero=is_mask, trainable=False)
        self.rnn = tf.keras.layers.SimpleRNN(units=units, activation='tanh')
        self.dense = tf.keras.layers.Dense(units=1)
    
    def call(self, inputs):
        #x = self.flatten(inputs)
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.dense(x)
        output = tf.nn.sigmoid(x)

        return output
