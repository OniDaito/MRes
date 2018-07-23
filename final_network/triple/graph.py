"""
graph.py - A bidirectional LSTM graph for triples
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
  initial = tf.truncated_normal(shape, mean = -0.01, stddev=0.001, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(-0.1, shape=shape, name=name)
  return tf.Variable(initial)

def lstm_cell(size, kprob, name):
  ''' Return an LSTM Cell or other RNN type cell. We
  have a few choices. We can even throw in a bit of
  dropout if we want.'''

  #cell = tf.nn.rnn_cell.BasicLSTMCell(size, name = name)
  #cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes = True, name = name)
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
  length = tf.cast(length, tf.int32)
  return length

# This is worth considering as dropout might help
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

def create_graph(FLAGS) :
  graph = tf.Graph()
  with tf.device(FLAGS.device):
    with graph.as_default():
      
      tf_train_dataset = None
      x = None
      input_size = 0
      hidden_size = 128
    
      if FLAGS.type_in == batcher.BatchTypeIn.FIVEDTRIPLE:
        ww = 15
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, ww],name="train_input") 
      elif FLAGS.type_in == batcher.BatchTypeIn.BITFIELDTRIPLE:
        ww = FLAGS.num_acids * 3
        x = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, ww],name="train_input") 
        
      dmask = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, 4], name="dmask") 

      # Since we are using dropout, we need to have a placeholder, so we dont set 
      # dropout at validation time
      keep_prob = tf.placeholder(tf.float32, name="keepprob")
      test = tf.placeholder(tf.float32, [None, FLAGS.max_cdr_length, 4], name="train_test")
     
      W_i = weight_variable([FLAGS.batch_size, input_size, hidden_size], "weight_i")
      b_i = bias_variable([hidden_size],"bias_i")
      output_i = tf.nn.tanh((tf.matmul(x, W_i) + b_i), name="output_i")
      #output_i = tf.tanh((tf.matmul(x, W_i)), name="output_i")
      #output_i = tf.nn.dropout(output_i, keep_prob)

      W_h = weight_variable([FLAGS.batch_size, hidden_size, FLAGS.max_cdr_length], "weight_h")
      b_h = bias_variable([FLAGS.max_cdr_length],"bias_h")
      output_h = tf.nn.relu((tf.matmul(output_i, W_h) + b_h), name="output_h")
      #output_h = tf.tanh((tf.matmul(output_i, W_h)), name="output_h")
      output_h = tf.nn.dropout(output_h, keep_prob)
  
      W_o = weight_variable([FLAGS.batch_size, FLAGS.max_cdr_length, 4], "weight_output")
      b_o = bias_variable([4],"bias_output")
      #output_o = tf.tanh((tf.matmul(output_h, W_o) + b_o) * dmask, name="output")
      output_o = tf.matmul(output_h, W_o, name="output") * dmask

      variable_summaries(output_o, "output_layer")
      variable_summaries(b_o, "output_bias")
      variable_summaries(W_o, "output_weight")

      variable_summaries(output_h, "hidden_layer")
      variable_summaries(b_h, "hidden_bias")
      variable_summaries(W_h, "hidden_weight")

      variable_summaries(output_i, "input_layer")
      variable_summaries(b_i, "input_bias")
      variable_summaries(W_i, "input_weight")


  return graph
