import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(3)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import sys
from simple_rnn_model import SimpleRNN
from dataloader import load_embeddings, load_dataset

import sys
sys.path.append("..")
from Tools.EmbdTFRecordMaker import read_embd_tfrecords

#param
embed_tfrecord = sys.argv[1]
tfrecordfile = sys.argv[2]
vocal_size = 1551856
embd_size = 50
units = 50


#load data
embd_table = read_embd_tfrecords(embed_tfrecord, [vocal_size, embd_size])
train_dataset = load_dataset(tfrecordfile)

#train param
num_epochs = 5
batch_size = 32
learning_rate = 0.001
is_mask = True
input_len = 10


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)
#batch_size = batch_size * strategy.num_replicas_in_sync

train_dataset = train_dataset.shuffle(buffer_size=1000)

train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
with strategy.scope():
    model = SimpleRNN(vocal_size=vocal_size, embd_size=embd_size, units=units, embd_table=embd_table, is_mask=is_mask)
    model.build(input_shape=(None, input_len))
    model.summary()
    model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['accuracy']
        )
model.fit(train_dataset, epochs=num_epochs)#, validation_data=train_dataset)
