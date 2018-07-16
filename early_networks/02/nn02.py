"""
nn02.py - Dealing with variable length input
author : Benjamin Blundell
email : me@benjamin.computer

Based on https://www.tensorflow.org/get_started/mnist/pros
and https://danijar.com/variable-sequence-lengths-in-tensorflow/

This version performs the best so far and is probably closest
to the TDNN we want to check

"""

import sys, os, math, random

import tensorflow as tf
import numpy as np

# Import our shared util
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
from common.util import *
from common.batch_real import *
from common import gen_data

FLAGS = NNGlobals()
# A higher learning rate seems good as we have few examples in this data set.
# Would that be correct do we think?
FLAGS.learning_rate = 0.35
FLAGS.window_size = 4
FLAGS.pickle_filename = 'pdb_martin_02.pickle'
FLAGS.num_epochs = 2000

def weight_variable(shape, name ):
  ''' TODO - we should check the weights here for our TDNN 
  For now I use truncated normals with stdddev of 0.1. Hopefully
  some of these go negative.'''
  initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(1.0, shape=shape, name=name)
  return tf.Variable(initial)

def conv1d(x, W):
  ''' Our convolution is what we use to replicate a TDNN though
  I suspect we need to do a lot more.'''
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def create_graph() :
  ''' My attempt at creating a TDNN with the conv1d operation. We have one conv layer, and two fully
  connected layers (which are quite large). We take a batch x max_cdr x amino_acid layer and output
  a max_cdr * 4 layer for angle components. We use tanh activation functions throughout.'''
  # Borrowed parts from https://stackoverflow.com/questions/41583540/custom-dropout-in-tensorflow#41584818
  graph = tf.Graph()

  with tf.device('/gpu:0'):
    with graph.as_default():
      # Input data - we use a 2D array with each 'channel' being an amino acid bitfield
      # In this case, we only use one example at a time as each is a different length
      tf_train_dataset = tf.placeholder(tf.bool, 
          [None, FLAGS.max_cdr_length, FLAGS.num_acids],name="train_input") 
      output_size = FLAGS.max_cdr_length * 4
      dmask = tf.placeholder(tf.float32, [None, output_size], name="dmask")
      x = tf.cast(tf_train_dataset, dtype=tf.float32)
      
      # According to the Tensorflow tutorial, the last two vars are input channels
      # and output channels (both 21)
      W_conv0 = weight_variable([FLAGS.window_size, 
        FLAGS.num_acids, FLAGS.num_acids] , "weight_conv_0")
      b_conv0 = bias_variable([FLAGS.num_acids], "bias_conv_0")
      
      # Using tanh as an activation fuction as it is bounded over -1 to 1
      # Don't have to use it here but we get better accuracy
      h_conv0 = tf.tanh(conv1d(x, W_conv0) + b_conv0)
  
      # The second layer is fully connected, neural net.
      dim_size = FLAGS.num_acids * FLAGS.max_cdr_length
      W_f = weight_variable([dim_size, output_size], "weight_hidden")
      b_f = bias_variable([output_size], "bias_hidden")
      

      # Apparently, the convolutional layer needs to be reshaped
      # This bit might be key as our depth, our 21 amino acid neurons are being connected here
      h_conv0_flat = tf.reshape(h_conv0, [-1, dim_size])
      h_f = tf.tanh( (tf.matmul(h_conv0_flat, W_f) + b_f)) * dmask
      
      # It looks like I can't take a size < max_cdr and use it, because we have 
      # fixed sized stuff so we need to dropout the weights we don't need per sample  
      # Find the actual sequence length and only include up to that length
      # We always use dropout even after training
      # Annoyingly tensorflow's dropout doesnt work for us here so I need to 
      # add another variable to our h_f layer, deactivating these neurons matched
      test = tf.placeholder(tf.float32, [None, output_size], name="train_test")

      # Output layer - we don't need this because the previous layer is fine but
      # we do get some accuracy increases with another layer/
      # the right number of variables for us but I'll add another one anyway so we have three.
      W_o = weight_variable([output_size, output_size], "weight_output")
      b_o = bias_variable([output_size],"bias_output")

      # I use tanh to bound the results between -1 and 1
      y_conv = tf.tanh( ( tf.matmul(h_f, W_o) + b_o) * dmask, name="output")
      variable_summaries(y_conv, "y_conv")

  return graph

def create_mask(batch):
  ''' create a mask for our fully connected layer, which
  is a [1] shape that is max_cdr * 4 long.'''
  mask = []
  for model in batch:
    mm = []
    for cdr in model:
      tt = 1
      if not 1 in cdr:
        tt = 0
      for i in range(0,4):
        mm.append(tt)
    mask.append(mm)
  return np.array(mask,dtype=np.float32)
    
def cost(goutput, gtest):
  ''' Our error function which we will try to minimise'''
  # We find the absolute difference between the output angles and the training angles
  # Can't use cross entropy because thats all to do with probabilities and the like
  # Basic error of sum squares diverges to NaN due to gradient so I go with reduce mean
  # Values of -3.0 are the ones we ignore
  # This could go wrong as adding 3.0 to -3.0 is not numerically stable
  mask = tf.sign(tf.add(gtest,3.0))
  basic_error = tf.square(gtest-goutput) * mask
  
  # reduce mean doesnt work here as we just want the numbers where mask is 1
  # We work out the mean ourselves
  basic_error = tf.reduce_sum(basic_error)
  basic_error /= tf.reduce_sum(mask)
  return basic_error

def run_session(graph, datasets):
  ''' Run the session once we have a graph, training methodology and a dataset '''
  with tf.device('/gpu:0'):
    with tf.Session(graph=graph) as sess:
      training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
      # Pull out the bits of the graph we need
      ginput = graph.get_tensor_by_name("train_input:0")
      gtest = graph.get_tensor_by_name("train_test:0")
      goutput = graph.get_tensor_by_name("output:0")
      gmask = graph.get_tensor_by_name("dmask:0")
      stepnum = 0
      # Working out the accuracy
      basic_error = cost(goutput, gtest) 
      # Setup all the logging for tensorboard 
      variable_summaries(basic_error, "Error")
      merged = tf.summary.merge_all() 
      train_writer = tf.summary.FileWriter('./summaries/train',graph)
      # So far, I have found Gradient Descent still wins out at the moment
      # https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
      train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(basic_error)
      #train_step = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(basic_error) 
      #train_step = tf.train.AdamOptimizer(1e-4).minimize(basic_error)
      #train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.1).minimize(basic_error)
      tf.global_variables_initializer().run()
      print('Initialized')	

      for i in range(0,FLAGS.num_epochs):
        stepnum = 0
        FLAGS.next_batch = 0
        print("Epoch",i)

        while has_next_batch(training_input, FLAGS):
          item_is, item_os = next_batch(training_input, training_output, FLAGS)
          mask = create_mask(item_is)
          summary, _ = sess.run([merged, train_step],
              feed_dict={ginput: item_is, gtest: item_os, gmask: mask})
          
          # Find the accuracy at every step, but only print every 100
          mask = create_mask(validate_input)
          train_accuracy = basic_error.eval(
              feed_dict={ginput: validate_input, gtest: validate_output,  gmask : mask}) 
          
          if stepnum % 50 == 0:
            print('step %d, training accuracy %g' % (stepnum, train_accuracy))
          
          #dm = gmask.eval(feed_dict={ginput: item_is, gtest: item_os, gmask: mask}) 
          #print(dm)
          stepnum += 1

      # save our trained net
      saver = tf.train.Saver()
      saver.save(sess, 'saved/nn02')

def run_saved(datasets):
  ''' Load the saved version and then test it against the validation set '''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn02.meta')
    saver.restore(sess, 'saved/nn02')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    mask = create_mask(validate_input)
    res = sess.run([goutput], feed_dict={ginput: validate_input, gmask: mask })

    # Now lets output a random example and see how close it is, as well as working out the 
    # the difference in mean values. Don't adjust the weights though
    r = random.randint(0, len(validate_input)-1)

    print("Actual              Predicted")
    for i in range(0,len(validate_input[r])):
      sys.stdout.write(bitmask_to_acid(FLAGS, validate_input[r][i]))
      phi = math.degrees(math.atan2(validate_output[r][i*4], validate_output[r][i*4+1]))
      psi = math.degrees(math.atan2(validate_output[r][i*4+2], validate_output[r][i*4+3]))
      sys.stdout.write(": " + "{0:<8}".format("{0:.3f}".format(phi)) + " ")
      sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi)) + " ")
      phi = math.degrees(math.atan2(res[0][r][i*4], res[0][r][i*4+1]))
      psi = math.degrees(math.atan2(res[0][r][i*4+2], res[0][r][i*4+3]))
      sys.stdout.write(" | " + "{0:<8}".format("{0:.3f}".format(phi)) + " ")
      sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi)))  
      print("")

def print_error(datasets):
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn02.meta')
    saver.restore(sess, 'saved/nn02')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gtest = graph.get_tensor_by_name("train_test:0")
    mask = create_mask(test_input)
    basic_error = cost(goutput, gtest)
    test_accuracy = basic_error.eval(
              feed_dict={ginput: test_input, gtest: test_output,  gmask : mask}) 
          
    print ("Error on test set:", test_accuracy)

def generate_pdbs(datasets):
  ''' Load the saved version and write a set of PDBs of both the predicted
  and actual models.'''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn02.meta')
    saver.restore(sess, 'saved/nn02')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    mask = create_mask(test_input)
    res = sess.run([goutput], feed_dict={ginput: test_input, gmask: mask })

    for midx in range(0,len(test_input)):
      torsions_real = []
      torsions_pred = []
      residues = []

      # Put the data in the correct arrays for PDB printing
      for i in range(0,len(test_input[midx])):
        tres = bitmask_to_acid(FLAGS, test_input[midx][i])
        if tres == "***": break
        residues.append((tres,i)) # TODO i is not correct - we need the reslabel
        phi = math.atan2(test_output[midx][i*4], test_output[midx][i*4+1])
        psi = math.atan2(test_output[midx][i*4+2], test_output[midx][i*4+3])
        torsions_real.append([phi,psi])
        phi = math.atan2(res[0][midx][i*4], res[0][midx][i*4+1])
        psi = math.atan2(res[0][midx][i*4+2], res[0][midx][i*4+3])
        torsions_pred.append([phi,psi])

      torsions_pred[0][0] = 0.0
      torsions_real[0][0] = 0.0 
      torsions_pred[len(torsions_pred)-1][1] = 0.0
      torsions_real[len(torsions_real)-1][1] = 0.0
      
      from common import torsion_to_coord as tc
  
      mname = str(midx).zfill(3) + "_real.pdb"
      with open(mname,'w') as f: 
        pf = {}
        pf["angles"] = torsions_real
        pf["residues"] = residues
        entries = tc.process(pf)
        f.write(tc.printpdb(mname, entries, residues))

      mname = str(midx).zfill(3) + "_pred.pdb"
      with open(mname,'w') as f: 
        pf = {}
        pf["angles"] = torsions_pred
        pf["residues"] = residues
        entries = tc.process(pf)
        f.write(tc.printpdb(mname, entries, residues))

      mname = str(midx).zfill(3) + "_real.txt"
      with open(mname,'w') as f: 
        for i in range(0, len(residues)):
          f.write(residues[i][0] + ": " + str(torsions_real[i][0]) + ", " + str(torsions_real[i][1]) + "\n")  

      mname = str(midx).zfill(3) + "_pred.txt"
      with open(mname,'w') as f: 
        for i in range(0, len(residues)):
          f.write(residues[i][0] + ": " + str(torsions_pred[i][0]) + ", " + str(torsions_pred[i][1]) + "\n")  

if __name__ == "__main__":
  from common import gen_data
  # If we just want to run the trained net
  datasets = init_data_sets(FLAGS, gen_data)
  
  if len(sys.argv) > 1:
    if sys.argv[1] == "-r":
      run_saved(datasets)
      sys.exit()
    
    elif sys.argv[1] == "-e":
      print_error(datasets)
      sys.exit()

    elif sys.argv[1] == "-g":
      generate_pdbs(datasets)
      sys.exit()

  graph = create_graph()
  run_session(graph, datasets)
  run_saved(datasets)
