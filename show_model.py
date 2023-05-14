import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os, sys
import numpy as np

from tensorflow.python.layers import base
import tensorflow as tf
import tensorflow.contrib.slim as slim
n_steps = 120
n_input = 36
n_classes = 11
n_hidden = 30
_X = tf.placeholder(tf.float32, [None, n_steps, n_input])
_weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
_biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
_X = tf.reshape(_X, [-1, n_input])   
_X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
_X = tf.split(_X, n_steps, 0) 

stack_cells_fw = []
stack_cells_bw = []
for i in range(3):
    stack_cells_fw.append(tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True))
    stack_cells_bw.append(tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True))
multi_cells_fw = tf.contrib.rnn.MultiRNNCell(stack_cells_fw)
multi_cells_bw = tf.contrib.rnn.MultiRNNCell(stack_cells_bw)
lstm_cells = tf.contrib.rnn.MultiRNNCell([multi_cells_fw, multi_cells_bw], state_is_tuple=True)
outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()
