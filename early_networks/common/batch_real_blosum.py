"""
batch_real_blosum.py - data as blosum matrix
author : Benjamin Blundell
email : oni@section9.co.uk
"""

import os, math, pickle
import numpy as np
import tensorflow as tf

from random import randint

# 20 x 20
lookup_table = {}
lookup_table["ALA"] = [5,-2,-2,-3,-1,-1,-1,0,-2,-2,-2,-1,-2,-3,-1,1,0,-4,-3,-1]
lookup_table["ARG"] = [-2,6,-1,-3,-5,1,-1,-3,0,-4,-3,2,-2,-4,-3,-1,-2,-4,-3,-3]
lookup_table["ASN"] = [-2,-1,7,1,-4,0,-1,-1,0,-4,-4,0,-3,-4,-3,0,0,-5,-3,-4]
lookup_table["ASP"] = [-3,-3,1,7,-5,-1,1,-2,-2,-5,-5,-1,-4,-5,-3,-1,-2,-6,-4,-5]
lookup_table["CYS"] = [-1,-5,-4,-5,9,-4,-6,-4,-5,-2,-2,-4,-2,-3,-4,-2,-2,-4,-4,-2]
lookup_table["GLN"] = [-1,1,0,-1,-4,7,2,-3,1,-4,-3,1,0,-4,-2,-1,-1,-3,-3,-3]
lookup_table["GLU"] = [-1,-1,-1,1,-6,2,6,-3,-1,-4,-4,0,-3,-5,-2,-1,-1,-5,-4,-3]
lookup_table["GLY"] = [0,-3,-1,-2,-4,-3,-3,6,-3,-5,-5,-2,-4,-5,-3,-1,-3,-4,-5,-5]
lookup_table["HIS"] = [-2,0,0,-2,-5,1,-1,-3,8,-4,-4,-1,-3,-2,-3,-2,-2,-3,1,-4]
lookup_table["ILE"] = [-2,-4,-4,-5,-2,-4,-4,-5,-4,5,1,-4,1,-1,-4,-3,-1,-4,-2,3]
lookup_table["LEU"] = [-2,-3,-4,-5,-2,-3,-4,-5,-4,1,5,-3,2,0,-4,-3,-2,-3,-2,0]
lookup_table["LYS"] = [-1,2,0,-1,-4,1,0,-2,-1,-4,-3,6,-2,-4,-2,-1,-1,-5,-3,-3]
lookup_table["MET"] = [-2,-2,-3,-4,-2,0,-3,-4,-3,1,2,-2,7,-1,-3,-2,-1,-2,-2,0]
lookup_table["PHE"] = [-3,-4,-4,-5,-3,-4,-5,-5,-2,-1,0,-4,-1,7,-4,-3,-3,0,3,-2]
lookup_table["PRO"] = [-1,-3,-3,-3,-4,-2,-2,-3,-3,-4,-4,-2,-3,-4,8,-2,-2,-5,-4,-3]
lookup_table["SER"] = [1,-1,0,-1,-2,-1,-1,-1,-2,-3,-3,-1,-2,-3,-2,5,1,-4,-3,-2]
lookup_table["THR"] = [0,-2,0,-2,-2,-1,-1,-3,-2,-1,-2,-1,-1,-3,-2,1,6,-4,-2,-1]
lookup_table["TRP"] = [-4,-4,-5,-6,-4,-3,-5,-4,-3,-4,-3,-5,-2,0,-5,-4,-4,11,2,-3]
lookup_table["TYR"] = [-3,-3,-3,-4,-4,-3,-4,-5,1,-2,-2,-3,-2,3,-4,-3,-2,2,8,-3]
lookup_table["VAL"] = [-1,-3,-4,-5,-2,-3,-3,-5,-4,3,0,-3,0,-2,-3,-2,-1,-3,-3,5]

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
    # We keep all our training input data in a single array training_size * (max_cdr * 5)
    # Output array is another large array training_size * ( max_cdr * 4 )
    # Input is a 2D array and the output is 2D
  
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

def vector_to_acid(FLAGS, vec):
  final = ""
  diff = 1000000
  if sum(vec) == 0:
    return "***"
  for aa in lookup_table.keys():
    tt = 0
    for i in range(0,len(vec)):
      tt += math.fabs(vec[i] - lookup_table[aa][i])

    if tt < diff:
      diff = tt
      final = aa

  return final

def has_next_batch(input_data, FLAGS):
  s = FLAGS.next_batch
  v = FLAGS.batch_size
  if FLAGS.next_batch + v >= len(input_data):
    return False
  return True

def next_batch(input_data, output_data, FLAGS) : 
  ''' Rather than use all the data at once, we batch it up instead.
  We can then get away with a 2D tensor at each step. We assume the 
  standard 28 * 20 size in and 124 size out.'''
  s = FLAGS.next_batch
  v = FLAGS.batch_size

  # By creating a new set of params, we avoid the NaN problem
  # we had with the larger sets
  # This does mean this only works with specific sizes at the moment
  ix = np.zeros((v, FLAGS.max_cdr_length, 20))
  ox = np.zeros((v, FLAGS.max_cdr_length * 4))

  for b in range(v):
    ix[b] = input_data[s + b]
    ox[b] = output_data[s + b]

  #print(str(FLAGS.next_batch) + " -> " + str(b))
  FLAGS.next_batch += v
  if FLAGS.next_batch >= len(input_data):
    FLAGS.next_batch  = len(input_data)

  return (ix, ox)

def random_batch(input_data, output_data, FLAGS) : 
  ''' A random set of length batch_size from the
  input data.'''
  
  ix = np.zeros((FLAGS.batch_size, FLAGS.max_cdr_length, 20))
  ox = np.zeros((FLAGS.batch_size, FLAGS.max_cdr_length * 4))

  b = 0
  for i in range(0,FLAGS.batch_size):
    rnd = randint(0,len(input_data)-1) 
    ix[b] = input_data[rnd]
    ox[b] = output_data[rnd]
    b += 1

  return (ix, ox)

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

    for i in range(0, FLAGS.max_cdr_length):
      tt.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      
    for i in range(0, FLAGS.max_cdr_length * 4):
      tu.append(-3.0)
  
    idx = 0
    for residue in aa['residues']:
      print(residue[0],idx)
      tt[idx] = lookup_table[residue[0]]
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
  input_set = np.array(ft,dtype=np.float32)
  output_set = np.array(fu,dtype=np.float32)

  print(input_set[1])

  return (input_set, output_set)
 
