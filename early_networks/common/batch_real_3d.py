"""
batch_real_3d.py - converting and batching our data for the real numbered angles
author : Benjamin Blundell
email : oni@section9.co.uk
"""

import os, math, pickle
import numpy as np
import tensorflow as tf

from random import randint

lookup_table = {}
lookup_table["ALA"] = [0.189, -3.989, 1.989, 0.14, 1.009]
lookup_table["ARG"] = [5.007, 0.834, -2.709, -2.027, 3.696]
lookup_table["ASN"] = [7.616, 0.943, 0.101, 3.308, 0.207]
lookup_table["ASP"] = [7.781, 0.03, 1.821, 1.376, -3.442]
lookup_table["CYS"] = [-5.929, -4.837, 6.206, 2.884, 5.365]
lookup_table["GLN"] = [5.48, 1.293, -3.091, -2.348, 1.628]
lookup_table["GLU"] = [7.444, 1.005, -2.121, -1.307, -1.011]
lookup_table["GLY"] = [4.096, 0.772, 7.12, 0.211, -1.744]
lookup_table["HIS"] = [3.488, 6.754, -2.703, 4.989, 0.452]
lookup_table["ILE"] = [-7.883, -4.9, -2.23, 0.99, -2.316]
lookup_table["LEU"] = [-7.582, -3.724, -2.74, -0.736, -0.208]
lookup_table["LYS"] = [5.665, -0.166, -2.643, -2.808, 2.474]
lookup_table["MET"] = [-5.2, -2.547, -3.561, -1.73, 0.859]
lookup_table["PHE"] = [-8.681, 4.397, -0.732, 1.883, -1.987]
lookup_table["PRO"] = [4.281, -2.932, 2.319, -3.269, -4.451]
lookup_table["SER"] = [4.201, -1.948, 1.453, 1.226, 1.014]
lookup_table["THR"] = [0.774, -3.192, 0.666, 0.07, 0.407]
lookup_table["TRP"] = [-8.492, 9.958, 4.874, -5.288, 0.672]
lookup_table["TYR"] = [-6.147, 7.59, -2.065, 2.413, -0.562]
lookup_table["VAL"] = [-6.108, -5.341, -1.953, 0.025, -2.062]

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

    # Write out the model names
    with open("model_mapping.txt","w") as f:
      idx = 0 
      f.write("***training set***\n")
      for t in training_angles.keys():
        f.write(str(idx) + "," + t + "\n")
        idx+=1
      idx = 0
      f.write("***test set***\n")
      for t in test_angles.keys():
        f.write(str(idx) + "," + t + "\n")
        idx+=1
      idx = 0
      f.write("***validate set***\n")
      for t in validate_angles.keys():
        f.write(str(idx) + "," + t + "\n")
        idx+=1

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
  standard 31 * 5 size in and 124 size out.'''
  s = FLAGS.next_batch
  v = FLAGS.batch_size

  # By creating a new set of params, we avoid the NaN problem
  # we had with the larger sets
  # This does mean this only works with specific sizes at the moment
  ix = np.zeros((v,28, 5))
  ox = np.zeros((v,28 * 4))

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
  
  ix = np.zeros((FLAGS.batch_size,28,5))
  ox = np.zeros((FLAGS.batch_size,28 * 4))

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

    for i in range(0,FLAGS.max_cdr_length):
      tt.append([0,0,0,0,0])
      
    for i in range(0,FLAGS.max_cdr_length * 4):
      tu.append(-3.0)
  
    idx = 0
    for residue in aa['residues']:
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
 
