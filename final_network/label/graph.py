"""
graph.py - Our labelling network
author : Benjamin Blundell
email : me@benjamin.computer

"""

import sys, os, math, random
import tensorflow as tf
import numpy as np

# Import our shared util - bit hacky but allows testing with __main__
if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  import common.acids as acids
  import common.batcher as batcher
  import common.settings
  from common.util_neural import *

def weight_variable(shape, name ):
  ''' For now I use truncated normals with stdddev of 0.1.'''
  initial = tf.truncated_normal(shape, stddev=0.01, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(0.0, shape=shape, name=name)
  return tf.Variable(initial)

def lstm_cell(size, kprob, name):
  ''' Return an LSTM Cell or other RNN type cell. We
  have a few choices. We can even throw in a bit of
  dropout if we want.'''

  #cell = tf.nn.rnn_cell.BasicLSTMCell(size, name = name)
  #cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes = True, name = name, activation=tf.nn.elu)
  #wrapper = tf.contrib.rnn.LSTMBlockWrapper
  cell = tf.nn.rnn_cell.GRUCell(size, name=name)
  #cell = tf.nn.rnn_cell.BasicRNNCell(size)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=kprob, output_keep_prob=kprob)
  return cell

def last_relevant(FLAGS, output, length, name="last"):
  ''' Taken from https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Essentially, we want the last output after the total CDR has been computed.'''
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * FLAGS.max_cdr_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index, name=name)
  return relevant

def create_length(batch):
  ''' return the actual lengths of our CDR here. Taken from
  https://danijar.com/variable-sequence-lengths-in-tensorflow/ '''
  used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32, name="length")
  return length

# This is worth considering as dropout might help
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

def create_graph(FLAGS) :
  graph = tf.Graph()
  with tf.device(FLAGS.device):
    with graph.as_default():
      
      x = None 
      ww = FLAGS.num_acids
      
      if FLAGS.type_in == batcher.BatchTypeIn.BITFIELD:
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, FLAGS.num_acids],name="train_input") 
      elif FLAGS.type_in == batcher.BatchTypeIn.FIVED or FLAGS.type_in == batcher.BatchTypeIn.FIVEDADD:
        ww = 5
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, ww],name="train_input") 
      elif FLAGS.type_in == batcher.BatchTypeIn.FIVEDTRIPLE:
        ww = 15
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, ww],name="train_input") 
      elif FLAGS.type_in == batcher.BatchTypeIn.BITFIELDTRIPLE:
        ww = FLAGS.num_acids * 3
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, ww],name="train_input") 
      
      num_classes = 36 * 36 # 10 degree divisions
      dmask = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, num_classes], name="dmask")  
      keep_prob = tf.placeholder(tf.float32, name="keepprob")
     
      #single_rnn_cell_fw = lstm_cell(FLAGS.lstm_size, keep_prob, "cell_fw")
      #single_rnn_cell_bw = lstm_cell(FLAGS.lstm_size, keep_prob, "cell_bw")

      sizes = [FLAGS.lstm_size,int(math.floor(FLAGS.lstm_size/2)),int(math.floor(FLAGS.lstm_size/4))]
      #sizes = [FLAGS.lstm_size, int(math.floor(FLAGS.lstm_size/2))]
      #sizes = [FLAGS.lstm_size, FLAGS.lstm_size]

      single_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell( [lstm_cell(sizes[i], keep_prob, "cell_fw" + str(i)) for i in range(len(sizes))])
      single_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell( [lstm_cell(sizes[i], keep_prob, "cell_bw" + str(i)) for i in range(len(sizes))])

      # 'outputs' is a tensor of shape [batch_size, max_cdr_length, lstm_size]
      # 'state' is a N-tuple where N is the number of LSTMCells containing a
      # tf.contrib.rnn.LSTMStateTuple for each cell
      length = create_length(x)
      initial_state = single_rnn_cell_fw.zero_state(FLAGS.batch_size, dtype=tf.float32)
      initial_state = single_rnn_cell_bw.zero_state(FLAGS.batch_size, dtype=tf.float32)
      
      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=single_rnn_cell_fw,cell_bw=single_rnn_cell_bw, inputs=x, dtype=tf.float32, sequence_length = length)
     
      #output, _ = tf.nn.dynamic_rnn(single_rnn_cell_fw, inputs=x, dtype=tf.float32, sequence_length = length)
      
      output_fw, output_bw = outputs
      states_fw, states_bw = states

      # We can avoid the costly gather operation here by using the state
      # Seems to only apply for single layer LSTMS potentially?
      #output_fw = last_relevant(FLAGS, output_fw, length, "last_fw")
      #output_bw = last_relevant(FLAGS, output_bw, length, "last_bw")  
  
      #output_fw = states_fw[-1]
      #output_bw = states_bw[-1]

      output = tf.concat((output_fw, output_bw), axis=2, name='bidirectional_concat_outputs')    
      #output = tf.add(output_fw, output_bw)
      output = tf.nn.dropout(output, keep_prob)

      #output = tf.reshape(output, [-1,sizes[-1]*2])
      #output /= 2.0

      dim = sizes[-1] * 2
      #dim = sizes[-1]

      # https://gist.github.com/danijar/d11c77c5565482e965d191929104440
      W_f = weight_variable([dim, num_classes], "weight_output")
      b_f = bias_variable([num_classes], "bias_output")

      # Flatten to apply same weights to all time steps.
      output = tf.reshape(output, [-1, dim])
      logits = tf.add(tf.matmul(output, W_f), b_f, name="logits")
      #logits = tf.matmul(output, W_f, name="logits")

      prediction = tf.reshape(tf.nn.softmax(logits), [-1, FLAGS.max_cdr_length, num_classes], name="prediction") 
      output = tf.reshape(logits, [-1, FLAGS.max_cdr_length, num_classes], name="output")
      
      test = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, num_classes], name="train_test")
      labels = tf.placeholder(tf.int32, [None, FLAGS.max_cdr_length], name="labels")
     
      variable_summaries(prediction, "output")
      variable_summaries(output, "mid_output")
      variable_summaries(W_f, "weight_output")
      variable_summaries(b_f, "bias_output")

  return graph
