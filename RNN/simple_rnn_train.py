import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import sys
from simple_rnn_model import SimpleRNN
from TFRecordMaker import read_tfrecords

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

embd_table = np.random.rand(20, 50)
#print(embd_table)



num_epochs = 5
batch_size = 32
learning_rate = 0.001

train_dataset = read_tfrecords(sys.argv[1])



optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)
batch_size = batch_size * strategy.num_replicas_in_sync

train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

with strategy.scope():
    model = SimpleRNN(vocal_size=20, embd_size=50, units=50, embd_table=embd_table, is_mask=True)
    model.build(input_shape=(None, 10))
    model.summary()
    model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['accuracy']
        )
model.fit(train_dataset, epochs=num_epochs)
