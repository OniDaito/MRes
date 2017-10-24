"""
nn01.py - First bash at the NN 
author : Benjamin Blundell
email : oni@section9.co.uk

Based on https://www.tensorflow.org/get_started/mnist/pros

"""

import sys, os, math, random

import tensorflow as tf
import numpy as np

from util import * 

FLAGS = NNGlobals()
FLAGS.learning_rate = 0.5
FLAGS.batch_size = 1
FLAGS.window_size = 8

# Import common items
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
from common import gen_data

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  Taken from https://www.tensorflow.org/get_started/summaries_and_tensorboard """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape, name ):
  ''' TODO - we should check the weights here for our TDNN '''
  initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)

def conv1d(x, W):
  ''' Our convolution is what we use to replicate a TDNN though
  I suspect we need to do a lot more.'''
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def create_graph() :
  ''' Create the tensorflow graph. This isnt a TDNN. Rather it's a covnet
  that has 21 channels, produces 32 feature maps and then recombines them
  into the final max_cdr * 4 output angles with a relu. Window size is 4.'''
  graph = tf.Graph()

  with tf.device('/gpu:0'):
    with graph.as_default():
      print("Creating graph")
      # Input data - we use a 2D array with each 'channel' being an amino acid bitfield
      # I'm hoping that we can set certain neurons to be sensitive to particular acid and it's appearance in time
      tf_train_dataset = tf.placeholder(tf.bool, [None, FLAGS.max_cdr_length, FLAGS.num_acids],name="train_input")
      output_size = FLAGS.max_cdr_length * 4

      # conversion from bool to something better
      x = tf.cast(tf_train_dataset,dtype=tf.float32)

      # We shall have a sliding window 4 long initially
      # TODO - are we sure these last two weights are in the right order / correct
      # According to the Tensorflow tutorial, the last two vars are input channels and output channels (both 21)
      W_conv0 = weight_variable([FLAGS.window_size, FLAGS.num_acids, FLAGS.num_acids] , "weight_conv_0")
      b_conv0 = bias_variable([FLAGS.num_acids], "bias_conv_0")
      # Using tanh as an activation fuction as it is bounded over -1 to 1
      # We might not even need a transfer function
      h_conv0 = tf.tanh(conv1d(x, W_conv0) + b_conv0)
      #h_conv0 = conv1d(x, W_conv0) + b_conv0

      # The second layer is fully connected, neural net. I believe this is essentially *flat*
      # But ouputs the 4 * max_cdr_size
      dim_size = FLAGS.num_acids * FLAGS.max_cdr_length
      W_f = weight_variable([dim_size, output_size], "weight_hidden")
      b_f = bias_variable([output_size], "bias_hidden")

      # Apparently, the convolutional layer needs to be reshaped
      # This bit might be key as our depth, our 21 amino acid neurons are being connected here
      # TODO - I know we need a reshape but I don't understand it.
      # TODO - can we be sure this is *fully connected*?
      h_conv0_flat = tf.reshape(h_conv0, [-1, dim_size])
      h_f = tf.tanh(tf.matmul(h_conv0_flat, W_f) + b_f)
      #h_f = tf.matmul(h_conv0_flat, W_f) + b_f
      
      # Output layer - we might not actually need this because the previous layer outputs
      # the right number of variables for us but I'll add another one anyway so we have three.
      W_o = weight_variable([output_size, output_size], "weight_output")
      b_o = bias_variable([output_size],"bias_output")

      # TODO - Could we somehow set the biases artificially for neurons we dont want to count
      # if our CDR length is shorter than the max? We should probably do that.
      # I use tanh to bound the results between -1 and 1
      y_conv = tf.tanh(tf.matmul(h_f, W_o) + b_o, name="output")
      test = tf.placeholder(tf.float32, [None, output_size], name="train_test")

      # Tensor board stuff
      variable_summaries(y_conv)

  return graph


def run_session(graph, datasets):
  ''' Run the session once we have a graph, training methodology and a dataset '''
  with tf.device('/gpu:0'):
    with tf.Session(graph=graph) as sess:
      tf.global_variables_initializer().run()
      print('Initialized')	
      training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
      # Pull out the bits of the graph we need
      ginput = graph.get_tensor_by_name("train_input:0")
      gtest = graph.get_tensor_by_name("train_test:0")
      goutput = graph.get_tensor_by_name("output:0")
      FLAGS.next_batch = 0
      stepnum = 0

      # Working out the accuracy
      # We find the absolute difference between the output angles and the training angles
      # Can't use cross entropy because thats all to do with probabilities and the like
      # Basic error of sum squares diverges to NaN due to gradient so I go with reduce mean
      # Need to use abs as minus values are also valid.
      #basic_error = tf.reduce_sum(tf.square(gtest - goutput))
      basic_error = tf.reduce_mean(tf.square(gtest - goutput))
      
      # Setup all the logging for tensorboard 
      variable_summaries(basic_error)
      merged = tf.summary.merge_all() 
      train_writer = tf.summary.FileWriter('./summaries_01/train',graph)

      # TODO - for some reason, the AdamOptimizer just dies :/
      #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
      # TODO - adjusting the learning rate - what effect does this have?
      train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(basic_error)

      while FLAGS.next_batch < training_input.shape[0]:
        batch_is, batch_os = next_batch(training_input, training_output, FLAGS)
        summary, _ = sess.run([merged, train_step], feed_dict={ginput: batch_is, gtest: batch_os})
        
        # We can work out the accuracy at every step as this is only small
        # we use the validation set
        train_accuracy = basic_error.eval(feed_dict={ginput: validate_input, gtest: validate_output}) 
        print('step %d, training accuracy %g' % (stepnum, train_accuracy))
        train_writer.add_summary(summary, stepnum)
        stepnum += 1

      # save our trained net
      saver = tf.train.Saver()
      saver.save(sess, 'saved/nn01')

def run_saved(datasets):
  ''' Load the saved version and then test it against the validation set '''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn01.meta')
    saver.restore(sess, 'saved/nn01')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    res = sess.run([goutput], feed_dict={ginput: validate_input})

    # Now lets output a random example and see how close it is, as well as working out the 
    # the difference in mean values. Don't adjust the weights though
    r = random.randint(0, len(validate_input))

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

if __name__ == "__main__":

  if len(sys.argv) > 1:
    if sys.argv[1] == "-r":
      datasets = init_data_sets(FLAGS)
      run_saved(datasets)
      sys.exit()

  datasets = init_data_sets(FLAGS)
  graph = create_graph()
  run_session(graph, datasets)
  run_saved(datasets)
