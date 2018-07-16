"""
util.py - support functions for the nn
author : Benjamin Blundell
email : me@benjamin.computer
"""

import os, math, pickle
import numpy as np
import tensorflow as tf

from random import randint

# Import common items
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
from . import gen_data

class NNGlobals (object):
  def __init__(self):
    self.num_acids = 20
    self.max_cdr_length = 28
    self.amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO", "SER", "THR", "TYR", "TRP", "VAL"]
    self.amino_acid_letter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F","P", "S", "T", "Y", "W", "V"]
    self.next_batch = 0
    self.batch_size = 20
    self.window_size = 4
    self.num_epochs = 5
    self.pickle_filename = 'pdb_martin_01.pickle'
    self.learning_rate = 0.1
    self.bin_size = 5

def variable_summaries(var, label):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  Taken from https://www.tensorflow.org/get_started/summaries_and_tensorboard """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean-' + label, mean)
    tf.summary.tensor_summary(label,var)
    #with tf.name_scope('stddev-' + label):
    #  stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #tf.summary.scalar('stddev-' + label, stddev)
    tf.summary.scalar('max-' + label, tf.reduce_max(var))
    tf.summary.scalar('min-' + label, tf.reduce_min(var))
    tf.summary.histogram('histogram-' + label , var)

def amino_mask(label, FLAGS):
  ''' Create a one hot encoded bitfield affair for
  each amino acid.'''
  idx = 0
  for aa in FLAGS.amino_acid_bitmask:
    if aa == label:
      return idx
    idx +=1
  return -1

def bitmask_to_acid(FLAGS, mask):
  ''' bitmask back to the acid label '''
  for i in range(0,len(FLAGS.amino_acid_bitmask)):
    if mask[i] == True:
      return FLAGS.amino_acid_bitmask[i]
  return "***"

def bitmask_to_letter(FLAGS, mask):
  ''' bitmask back to the acid letter '''
  for i in range(0,len(FLAGS.amino_acid_letter)):
    if mask[i] == True:
      return FLAGS.amino_acid_letter[i]
  return "***"

def pickle_it(filename, dataset):
  ''' Save our datasets if we like. '''
  with open(filename, 'wb') as f:
    print("Pickling data...")
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
