"""
batch_discrete_3d.py - A dense representation for discrete angles
author : Benjamin Blundell
email : me@benjamin.computer
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
  ix = np.zeros((v,FLAGS.max_cdr_length, 5))
  ox = np.zeros((FLAGS.batch_size, FLAGS.max_cdr_length * 2, FLAGS.num_classes))

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
  We create a set of bins, one for each angle.'''
  ft = []
  fu = []
  
  for model in angle_set:
    aa = angle_set[model]
    tt = []
    tu = []

    bin_dim = int(math.floor(360 / FLAGS.bin_size))  # 30 degrees so 12 classes

    for i in range(0,FLAGS.max_cdr_length):
      tt.append([0 for x in range(0,5)])
      
    for i in range(0,FLAGS.max_cdr_length * 2):
      cmask = []
      for i in range(0,bin_dim):
        cmask.append(0.0)
      
      tu.append(cmask)
    
    idx = 0
    for residue in aa['residues']:
      tt[idx] = lookup_table[residue[0]]
      idx += 1

    # unpick the angle tuple
    idx = 0
    for angle in aa['angles']:
      phi = math.degrees(math.atan2(angle[0], angle[1]))
      psi = math.degrees(math.atan2(angle[2], angle[3]))

      phi_cls = int(math.floor( (phi + 180) / FLAGS.bin_size))
      psi_cls = int(math.floor( (psi + 180) / FLAGS.bin_size))

      tu[idx*2][phi_cls] = 1.0
      tu[idx*2+1][psi_cls] = 1.0
      idx += 1

    ft.append(tt)
    fu.append(tu)

  print(len(fu), len(fu[0]), len(ft), len(ft[1]))
  input_set = np.array(ft,dtype=np.float32)
  output_set = np.array(fu,dtype=np.float32) # You'd think an int32 here but no

  return (input_set, output_set)

def classmask_to_angles(mask, FLAGS):
  ''' Given a one-hot encoded input classmask, or the output softmax one, return 
  which bin it falls in and compute the angles. '''
  bin_dim = int(360 / FLAGS.bin_size)
  cls = 0
  ll = 0
  for i in range(0, bin_dim):
    if mask[i] >= ll:
      ll = mask[i]
      cls = i

  angle = FLAGS.bin_size * cls - 180
  return angle


