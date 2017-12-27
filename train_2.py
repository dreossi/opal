import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import models

# Adding seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

# Prepare input data
classes = ['bad','good']
num_classes = len(classes)

# 20% of the data will be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='./data/train/'
check_point_name = './pero-model'

# Load training and validation images and labels
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

# Labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



# Network graph params

filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

nn = models.mediumNN(x)

#y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred = tf.nn.softmax(nn.output,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

session.run(tf.global_variables_initializer())


# Training functions
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=nn.output, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    '''Show progress while training'''
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    '''Training loop'''

    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):

        # Fecth batch
        x_batch, y_true_batch, _, _ = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(batch_size)


        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        # Show progress and save learnt parameters
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, check_point_name)


    total_iterations += num_iteration

train(num_iteration=3000)
