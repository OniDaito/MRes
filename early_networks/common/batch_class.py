"""
batch_class.py - converting and batching data into classes
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
from .util import amino_mask, pickle_it


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

def has_next_batch(input_data, FLAGS):
  s = FLAGS.next_batch
  v = FLAGS.batch_size
  if FLAGS.next_batch + v >= len(input_data):
    return False
  return True


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

    bin_dim = 360 / FLAGS.bin_size
    total_bins = int(math.floor(math.pow(bin_dim,2)))

    for i in range(0,FLAGS.max_cdr_length):
      tt.append([False for x in range(0,FLAGS.num_acids)])
      
    for i in range(0,FLAGS.max_cdr_length):
      cmask = []
      for i in range(0,total_bins):
        cmask.append(0.0)
      
      tu.append(cmask)
  
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
      phi = math.degrees(math.atan2(angle[0], angle[1]))
      psi = math.degrees(math.atan2(angle[2], angle[3]))

      if psi == 180.0: psi = 179.0 # TODO - we really should ignore this last angle and the first

      # 30 degree increments for our classes, 0 to 5183 inclusive
      # We basically have a bitfield that is 5184 long!
      cls = int(((math.floor((psi + 180) / FLAGS.bin_size) * bin_dim) + math.floor((phi + 180) / FLAGS.bin_size)))
      tu[idx][cls] = 1.0
      idx += 1

    ft.append(tt)
    fu.append(tu)

  print(len(fu), len(fu[0]), len(ft), len(ft[1]))
  input_set = np.array(ft,dtype=np.bool)
  output_set = np.array(fu,dtype=np.float32) # You'd think an int32 here but no

  return (input_set, output_set)

def classmask_to_angles(mask, FLAGS):
  ''' Given a one-hot encoded input classmask, or the output softmax one, return 
  which bin it falls in and compute the angles. '''
  bin_dim = int(360 / FLAGS.bin_size)
  cls_size = int(math.pow(bin_dim,2))
  cls = 0
  ll = 0
  for i in range(0,cls_size):
    if mask[i] >= ll:
      ll = mask[i]
      cls = i

  phi = ((cls % bin_dim) * FLAGS.bin_size) - 180
  psi = (math.floor(cls / bin_dim) * FLAGS.bin_size) - 180
  return (phi,psi)


