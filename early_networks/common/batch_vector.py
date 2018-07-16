"""
batch_vector.py - converting and batching our data for the real numbered angles
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
from .util import amino_mask, pickle_it

def init_data_sets(FLAGS, gen_data) :
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
  
    training_input, training_output, training_lengths = angles_to_array(training_angles, FLAGS)
    validate_input, validate_output, validate_lengths= angles_to_array(validate_angles, FLAGS)
    test_input, test_output, test_lengths = angles_to_array(test_angles, FLAGS)
    # I wanna see you pickle it (just a little bit)
    pickle_it(FLAGS.pickle_filename,(training_input, training_output, training_lengths,validate_input, validate_output, validate_lengths, test_input, test_output, test_lengths))

  else:
    print("Loading from pickle file")
    with open(FLAGS.pickle_filename, 'rb') as f:
      training_input, training_output, training_lengths, validate_input, validate_output, validate_lengths, test_input, test_output, test_lengths = pickle.load(f)

  return (training_input, training_output, training_lengths, validate_input, validate_output, validate_lengths, test_input, test_output, test_lengths)


def has_next_batch(input_data, FLAGS):
  s = FLAGS.next_batch
  v = FLAGS.batch_size
  if FLAGS.next_batch + v >= len(input_data):
    return False
  return True

def next_batch(input_data, output_data, lengths, FLAGS) : 
  ''' Rather than use all the data at once, we batch it up instead.
  We can then get away with a 2D tensor at each step. We assume the 
  standard 28 * 20 size in and 84 size out.'''
  s = FLAGS.next_batch
  v = FLAGS.batch_size

  # By creating a new set of params, we avoid the NaN problem
  # we had with the larger sets
  # This does mean this only works with specific sizes at the moment
  ix = np.zeros((v,28, 20))
  ox = np.zeros((v,28 * 3))
  lx = np.zeros((v))

  for b in range(v):
    ix[b] = input_data[s + b]
    ox[b] = output_data[s + b]
    lx[b] = lengths[s+b]

  #print(str(FLAGS.next_batch) + " -> " + str(b))
  FLAGS.next_batch += v
  if FLAGS.next_batch >= len(input_data):
    FLAGS.next_batch  = len(input_data)

  return (ix, ox, lx)

def random_batch(input_data, output_data, lengths, FLAGS) : 
  ''' A random set of length batch_size from the
  input data.'''
  
  ix = np.zeros((FLAGS.batch_size,28,20))
  ox = np.zeros((FLAGS.batch_size,28 * 3))
  lx = np.zeros((FLAGS.batch_size))

  b = 0
  for i in range(0,FLAGS.batch_size):
    rnd = randint(0,len(input_data)-1) 
    ix[b] = input_data[rnd]
    ox[b] = output_data[rnd]
    lx[b] = lengths[rnd]
    b += 1

  return (ix, ox, lx)

def next_item(input_data, output_data, lengths, FLAGS) : 
  ''' In this case, we just return the next item in the 
  training and test sets.'''

  s = FLAGS.next_batch
  FLAGS.next_batch = s+1
  return (input_data[s:s+1], output_data[s:s+1])

def angles_to_array(angle_set, FLAGS):
  ''' Convert our sequence and angles to numpy arrays.
  We might be able to do more of this in SQL. Our arrays here
  are max_cdr x num_acids - 2D. We generate a feature map for
  each channel, each being an amino acid.'''
  ft = []
  fu = []
  lengths = []
  
  for model in angle_set:
    aa = angle_set[model]
    tt = []
    tu = []

    for i in range(0,FLAGS.max_cdr_length):
      tt.append([False for x in range(0,FLAGS.num_acids)])
      
    for i in range(0,FLAGS.max_cdr_length * 3):
      tu.append(1.0)
  
    idx = 0
    for residue in aa['residues']:
      ta = []

      for n in range(0, FLAGS.num_acids):
        ta.append(False)
  
      mask = amino_mask(residue[0], FLAGS) 
      ta[mask] = True
      tt[idx] = ta
      idx += 1

    lengths.append(len(aa['residues']))
  
    # unpick the angle tuple
    idx = 0
    for angle in aa['vectors']:
      tu[idx*3] = angle[0]
      tu[idx*3+1] = angle[1]
      tu[idx*3+2] = angle[2]
      idx += 1

    ft.append(tt)
    fu.append(tu)

  print(len(fu), len(fu[0]), len(ft), len(ft[1]))
  input_set = np.array(ft,dtype=np.bool)
  output_set = np.array(fu,dtype=np.float32)

  return (input_set, output_set, lengths)
