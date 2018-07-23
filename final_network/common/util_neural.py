"""
util_neural.py - support functions for the nn
author : Benjamin Blundell
email : me@benjamin.computer

"""
import tensorflow as tf
import numpy as np

def variable_summaries(var, label):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  Taken from https://www.tensorflow.org/get_started/summaries_and_tensorboard """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean-' + label, mean)
    tf.summary.tensor_summary(label,var)
    with tf.name_scope('stddev-' + label):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev-' + label, stddev)
    tf.summary.scalar('max-' + label, tf.reduce_max(var))
    tf.summary.scalar('min-' + label, tf.reduce_min(var))
    tf.summary.histogram('histogram-' + label , var)

