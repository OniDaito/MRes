"""
util.py - support functions for the nn
author : Benjamin Blundell
email : oni@section9.co.uk
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
    self.num_acids = 21
    self.max_cdr_length = 31
    self.amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO","SEC", "SER", "THR", "TYR", "TRP", "VAL"]
    self.next_batch = 0
    self.batch_size = 20
    self.window_size = 4
    self.num_epochs = 5
    self.pickle_filename = 'pdb_martin_01.pickle'
    self.learning_rate = 0.1

def variable_summaries(var, label):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  Taken from https://www.tensorflow.org/get_started/summaries_and_tensorboard """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean-' + label, mean)
    with tf.name_scope('stddev-' + label):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev-' + label, stddev)
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

def angles_to_array(angle_set, FLAGS):
  ''' Convert our sequence and angles to numpy arrays.
  We might be able to do more of this in SQL. Our arrays here
  are max_cdr x num_acids - 2D. We generate a feature map for
  each channel, each being an amino acid.'''
  ft = []
  fu = []
  
  for model in angle_set:
    aa = angle_set[model]
    tt = []
    tu = []

    for i in range(0,FLAGS.max_cdr_length):
      tt.append([False for x in range(0,FLAGS.num_acids)])
      
    for i in range(0,FLAGS.max_cdr_length * 4):
      tu.append(-3.0)
  
    idx = 0
    for residue in aa['residues']:
      ta = []

      for n in range(0, FLAGS.num_acids):
        ta.append(False)
  
      mask = amino_mask(residue[0], FLAGS) 
      ta[mask] = True
      tt[idx] = ta
      idx += 1

    # unpick the angle tuple
    idx = 0
    for angle in aa['angles']:
      tu[idx*4] = angle[0]
      tu[idx*4+1] = angle[1]
      tu[idx*4+2] = angle[2]
      tu[idx*4+3] = angle[3]
      idx += 1

    ft.append(tt)
    fu.append(tu)

  print(len(fu), len(fu[0]), len(ft), len(ft[1]))
  input_set = np.array(ft,dtype=np.bool)
  output_set = np.array(fu,dtype=np.float32)

  return (input_set, output_set)

def pickle_it(filename, dataset):
  ''' Save our datasets if we like. '''
  with open(filename, 'wb') as f:
    print("Pickling data...")
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def init_data_sets(FLAGS) :
  ''' Create the datasets we need for training and the like. If the data 
  is already there and pickled, use that to save time.'''

  training_input = training_output = validate_input = validate_output = test_input= test_output = None

  if not os.path.exists(FLAGS.pickle_filename):

    # Not the most memory efficient I guess but our dataset is small
    angles = gen_data.gen()
    training_angles, validate_angles, test_angles = gen_data.create_data_sets(angles)

    # Create the input tensors and the like - need to create a series of 1D Tensors for
    # input, a bitmask for our neural net
    
    # We keep all our training input data in a single array training_size * (max_cdr * num_acids)
    # Output array is another large array training_size * ( max_cdr * 4 )
    # Input is a 2D array and the output is 2D
    # We use an all blank array where amino acids don't exist
    # We could use a single bit to represent each amino acid, but instead im using a bool and 
    # hoping numpy does the decent thing. Not too short of memory in this project
  
    training_input, training_output = angles_to_array(training_angles, FLAGS)
    validate_input, validate_output = angles_to_array(validate_angles, FLAGS)
    test_input, test_output = angles_to_array(test_angles, FLAGS)
    # I wanna see you pickle it (just a little bit)
    pickle_it(FLAGS.pickle_filename,(training_input, training_output, validate_input, validate_output, test_input, test_output))

  else:
    print("Loading from pickle file")
    with open(FLAGS.pickle_filename, 'rb') as f:
      training_input, training_output, validate_input, validate_output, test_input, test_output = pickle.load(f)

  return (training_input, training_output, validate_input, validate_output, test_input, test_output)

def next_batch(input_data, output_data, FLAGS) : 
  ''' Rather than use all the data at once, we batch it up instead.
  We can then get away with a 2D tensor at each step '''

  s = FLAGS.next_batch
  b = s + FLAGS.batch_size
  if b > len(input_data):
    b = len(input_data)

  #print(str(FLAGS.next_batch) + " -> " + str(b))
  FLAGS.next_batch = b

  return (input_data[s:b], output_data[s:b])

def random_batch(input_data, output_data, FLAGS) : 
  ''' A random set of length batch_size from the
  input data.'''
  
  inputd = []
  outputd = []
  for i in range(0,FLAGS.batch_size):
    rnd = randint(0,len(input_data)-1) 
    inputd.append(input_data[rnd])
    outputd.append(output_data[rnd])

  return (inputd, outputd)

def next_item(input_data, output_data, FLAGS) : 
  ''' In this case, we just return the next item in the 
  training and test sets.'''

  s = FLAGS.next_batch
  FLAGS.next_batch = s+1
  return (input_data[s:s+1], output_data[s:s+1])


