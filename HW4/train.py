from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow and TensorFlow Datasets

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
import os
import time

# Important configuration
EFFECTIVE_BATCH_SIZE = 64

# Prepare the data
datasets, info = tfds.load(name='cifar10', with_info=True, as_supervised=True)

set_train, set_test = datasets['train'], datasets['test']

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE = EFFECTIVE_BATCH_SIZE

def scale(image, label):
    paddings = tf.constant([[2, 2,], [2, 2], [1, 1]])
    image = tf.pad(image, paddings, 'CONSTANT')

    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

train_dataset = set_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = set_test.map(scale).batch(BATCH_SIZE)

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(time.time() - self.timetaken)
        self.timetaken = time.time()
    def on_train_end(self,logs = {}):
        total_time = sum(self.times[1:])
        print("Training time:{0:.3f} seconds, Throughput: {1:.3f}, Training cost: {2:.3f}"
            .format(total_time, num_train_examples*4 / total_time, strategy.num_replicas_in_sync * total_time * 0.9 / 360))

# Define the model
with strategy.scope():
    model = tf.keras.applications.resnet.ResNet50(include_top=False, input_shape=(36, 36, 5), classes=10)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

timetaken = timecallback()
model.fit(train_dataset, epochs=5, callbacks = [timetaken])