"""
nn06.py - A bidirectional LSTM attempt
author : Benjamin Blundell
email : me@benjamin.computer

We pad out the data to the maximum, but for each input
we find the real length and both stop the LSTM unrolls
at that point (dynamic) and mask out the output layers
so the cost function doesn't take these padded values
into account. 

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
FLAGS.learning_rate = 0.45
FLAGS.pickle_filename = 'pdb_martin_06.pickle'
FLAGS.lstm_size = 256   # number of neurons per LSTM cell do we think? 
FLAGS.num_epochs = 2000 # number of loops around the training set
FLAGS.batch_size = 20

def weight_variable(shape, name ):
  ''' For now I use truncated normals with stdddev of 0.1.'''
  initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(1.0, shape=shape, name=name)
  return tf.Variable(initial)

def lstm_cell(size, kprob):
  ''' Return an LSTM Cell or other RNN type cell. We
  have a few choices. We can even throw in a bit of
  dropout if we want.'''

  cell= tf.nn.rnn_cell.BasicLSTMCell(size)
  #cell = tf.nn.rnn_cell.GRUCell(size)
  #cell = tf.nn.rnn_cell.BasicRNNCell(size)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=kprob)
  return cell

def create_graph() :
  graph = tf.Graph()

  with tf.device('/gpu:0'):
    with graph.as_default():
      # Input data. We take in padded CDRs but feed in a length / mask as well
      # Apparently the dynamic RNN thingy can cope with variable lengths
      # Input has to be [batch_size, max_time, ...]
      tf_train_dataset = tf.placeholder(tf.int32, [None, FLAGS.max_cdr_length, FLAGS.num_acids],name="train_input") 
      output_size = FLAGS.max_cdr_length * 4
      dmask = tf.placeholder(tf.float32, [None, output_size], name="dmask")
      x = tf.cast(tf_train_dataset, dtype=tf.float32)
      
      # Since we are using dropout, we need to have a placeholder, so we dont set 
      # dropout at validation time
      keep_prob = tf.placeholder(tf.float32, name="keepprob")

      # This is the number of unrolls I think - sequential cells
      # In this example, I'm going for max_cdr_length as we want all the history
      # This will take a while and it is dynamically sized based on the inputs.
      #sizes = []
      #for i in range(0,FLAGS.max_cdr_length):
      #  sizes.append(FLAGS.lstm_size)
      #rnn_layers = [lstm_cell(size, keep_prob) for size in sizes]
      #multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
      
      # TODO - So most examples appear to just use the ONE cell. Why is this? Do we
      # use the length bit for the number of unrolls? Have I been doing this wrong?
      
      single_rnn_cell = lstm_cell(FLAGS.lstm_size, keep_prob)
      # 'outputs' is a tensor of shape [batch_size, max_cdr_length, lstm_size]
      # 'state' is a N-tuple where N is the number of LSTMCells containing a
      # tf.contrib.rnn.LSTMStateTuple for each cell
      length = create_length(x)
      initial_state = single_rnn_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=single_rnn_cell,cell_bw=single_rnn_cell, inputs=x, dtype=tf.float32, sequence_length = length)
      
      output_fw, output_bw = outputs
      states_fw, states_bw = states

      # We flatten out the outputs so it just looks like a big batch to our weight matrix
      # apparently this gives us weights across the entire set of steps
      # TODO - this has resulted in worse performance in the past but we will go with it 
      # here because we are combining forward and backward passes
      output_fw = tf.reshape(output_fw, [-1, FLAGS.lstm_size], name="flattened_fw")
      output_bw = tf.reshape(output_bw, [-1, FLAGS.lstm_size], name="flattened_bw")

      output = tf.add(output_fw, output_bw) # TODO - adding? Really?

      test = tf.placeholder(tf.float32, [None, output_size], name="train_test")

      W_i = weight_variable([FLAGS.lstm_size, 4], "weight_intermediate")
      b_i = bias_variable([4],"bias_intermediate")
      y_i = tf.tanh( ( tf.matmul( output, W_i) + b_i), name="intermediate")

      # Now reshape it back and run the mask against it
      y_b = tf.reshape(y_i, [-1, output_size], name="output")
      y_b = y_b * dmask

      

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

def create_length(batch):
  ''' return the actual lengths of our CDR here. Taken from
  https://danijar.com/variable-sequence-lengths-in-tensorflow/ '''
  used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

def cost(goutput, gtest):
  ''' Our error function which we will try to minimise'''
  # TODO - could pass length in here perhaps instead of checking against -3.0
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

      gt = graph.get_tensor_by_name("weight_intermediate:0")
      print("weight_intermediate",gt.shape)

      gt = graph.get_tensor_by_name("intermediate:0")
      print("intermediate",gt.shape)

      #gt = graph.get_tensor_by_name("flattened:0")
      #print("flattened",gt.shape)

      #gt = graph.get_tensor_by_name("reshaped:0")
      #print("reshaped",gt.shape)


      gt = graph.get_tensor_by_name("output:0")
      print("output",gt.shape)

      #return

      training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
      # Pull out the bits of the graph we need
      ginput = graph.get_tensor_by_name("train_input:0")
      gtest = graph.get_tensor_by_name("train_test:0")
      goutput = graph.get_tensor_by_name("output:0")
      gmask = graph.get_tensor_by_name("dmask:0")
      gprob = graph.get_tensor_by_name("keepprob:0")

      # Working out the accuracy
      basic_error = cost(goutput, gtest) 
      # Setup all the logging for tensorboard 
      variable_summaries(basic_error, "Error")
      merged = tf.summary.merge_all() 
      train_writer = tf.summary.FileWriter('./summaries/train',graph)
      
      # So far, I have found Gradient Descent still wins out at the moment
      # https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
      
      #train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(basic_error)
      optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
      gvs = optimizer.compute_gradients(basic_error)
      capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      train_step = optimizer.apply_gradients(capped_gvs)
      
      #train_step = tf.train.AdamOptimizer(1e-4).minimize(basic_error)
      #train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.1).minimize(basic_error)
      
      tf.global_variables_initializer().run()
      print('Initialized')	

      for i in range(0,FLAGS.num_epochs):
        stepnum = 0
        FLAGS.next_batch = 0
        print("Epoch",i)

        while has_next_batch(training_input, FLAGS):
          batch_is, batch_os = next_batch(training_input, training_output, FLAGS)
          batch_iv, batch_ov = random_batch(validate_input, validate_output, FLAGS)
      
          # For some reason, if the batches are not ALL the same size, we get a crash
          # so I reject batches smaller than the one set
          # This seems to be due to the gradient clipping, whatever that is?
          if len(batch_is) != FLAGS.batch_size or len(batch_iv) != FLAGS.batch_size:
            continue

          mask = create_mask(batch_is)
          summary, _ = sess.run([merged, train_step],
              feed_dict={ginput: batch_is, gtest: batch_os, gmask: mask, gprob: 0.8})
          
          # Find the accuracy at every step, but only print every 100
          # We have to batch here too for some reason? LSTM or something?
          mask = create_mask(batch_iv)
          train_accuracy = basic_error.eval(
              feed_dict={ginput: batch_iv, gtest: batch_ov,  gmask: mask, gprob: 1.0}) 
          
          if stepnum % 10 == 0:
            print('step %d, training accuracy %g' % (stepnum, train_accuracy))
          
          #dm = gmask.eval(feed_dict={ginput: item_is, gtest: item_os, gmask: mask}) 
          #print(dm)
          train_writer.add_summary(summary, stepnum)
          stepnum += 1

      # save our trained net
      saver = tf.train.Saver()
      saver.save(sess, 'saved/nn06')

def run_saved(datasets):
  ''' Load the saved version and then test it against the validation set '''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn06.meta')
    saver.restore(sess, 'saved/nn06')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gprob = graph.get_tensor_by_name("keepprob:0")
    mask = create_mask(validate_input)
    res = sess.run([goutput], feed_dict={ginput: validate_input, gmask: mask, gprob: 1.0})

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
    saver = tf.train.import_meta_graph('saved/nn06.meta')
    saver.restore(sess, 'saved/nn06')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0") 
    gprob = graph.get_tensor_by_name("keepprob:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gtest = graph.get_tensor_by_name("train_test:0")
    mask = create_mask(test_input)
    basic_error = cost(goutput, gtest)

    # TODO - we should break this up and output
    test_input = test_input[:FLAGS.batch_size]
    test_output = test_output[:FLAGS.batch_size]

    test_accuracy = basic_error.eval(
        feed_dict={ginput: test_input, gtest: test_output, gmask : mask, gprob: 1.0}) 
          
    print ("Error on test set:", test_accuracy)

def generate_pdbs(datasets):
  ''' Load the saved version and write a set of PDBs of both the predicted
  and actual models.'''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph('saved/nn06.meta')
    saver.restore(sess, 'saved/nn06')
    training_input, training_output, validate_input, validate_output, test_input, test_output = datasets
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gprob = graph.get_tensor_by_name("keepprob:0")
    mask = create_mask(test_input)
    
    
    # TODO - we should break this up and output
    # We use batches here because this NN is awkward
    midx = 0

    for k in range(0, len(test_input), FLAGS.batch_size):
      test_input_batch = test_input[k:k+FLAGS.batch_size]
      test_output_batch = test_output[k:k+FLAGS.batch_size]
      
      if len(test_input_batch) != FLAGS.batch_size:
        break

      res = sess.run([goutput], feed_dict={ginput: test_input_batch, gmask: mask, gprob: 1.0 })

      for j in range(0,len(test_input_batch)):
        torsions_real = []
        torsions_pred = []
        residues = []

        # Put the data in the correct arrays for PDB printing
        for i in range(0,len(test_input_batch[j])):
          tres = bitmask_to_acid(FLAGS, test_input_batch[j][i])
          if tres == "***": break
          residues.append((tres,i)) # TODO i is not correct - we need the reslabel
          phi = math.atan2(test_output_batch[j][i*4], test_output_batch[j][i*4+1])
          psi = math.atan2(test_output_batch[j][i*4+2], test_output_batch[j][i*4+3])
          torsions_real.append([phi,psi])
          phi = math.atan2(res[0][j][i*4], res[0][j][i*4+1])
          psi = math.atan2(res[0][j][i*4+2], res[0][j][i*4+3])
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
        
        midx += 1

if __name__ == "__main__":
  from common import gen_data
  # If we just want to run the trained net
  if len(sys.argv) > 1:
    if sys.argv[1] == "-r":
      datasets = init_data_sets(FLAGS, gen_data)
      run_saved(datasets)
      sys.exit()
    if sys.argv[1] == "-e":
      datasets = init_data_sets(FLAGS, gen_data)
      print_error(datasets)
      sys.exit()
    if sys.argv[1] == "-g":
      datasets = init_data_sets(FLAGS, gen_data)
      generate_pdbs(datasets)
      sys.exit()

  datasets = init_data_sets(FLAGS, gen_data)
  graph = create_graph()
  run_session(graph, datasets)
  run_saved(datasets)
