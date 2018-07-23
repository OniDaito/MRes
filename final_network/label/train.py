"""
train.py - train our neural network
author : Benjamin Blundell
email : me@benjamin.computer

"""
import os, sys, math, random
import tensorflow as tf
import numpy as np

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.util_neural import *
  from common import acids
  from common import batcher
  from run import test


def cost(goutput, gtest, FLAGS):
  ''' Our error function which we will try to minimise''' 
  mask = tf.sign(tf.add(gtest,3.0))
  basic_error = tf.square(gtest-goutput) * mask
  basic_error = tf.reduce_sum(basic_error)
  basic_error /= tf.reduce_sum(mask)  
  #basic_error += 0.001*tf.nn.l2_loss(gweights) 
  return basic_error

def cost_cat(gpred, gtest, glength, FLAGS):
  ''' Our error function which we will try to minimise''' 
  #mask = tf.sign(tf.add(gtest,3.0))
  mask = tf.sign(tf.reduce_max(tf.abs(gtest),2))
  #mask = tf.sign(tf.reduce_max(tf.abs(gtest),2))
  
  cross_entropy = gtest * tf.log(gpred)
  cross_entropy = -tf.reduce_sum(cross_entropy, 2)
  cross_entropy *= mask
 
  cross_entropy = tf.reduce_sum(cross_entropy, 1)
  cross_entropy /= tf.cast(glength, tf.float32)
  cross_entropy = tf.reduce_mean(cross_entropy) 
  
  return cross_entropy

def cost_logit(logits, labels, length, FLAGS):
  # https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
  labels_flat = tf.reshape(labels, [-1])
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_flat) + 1e-10
  mask = tf.sign(tf.to_float(labels))
  mask = tf.reshape(mask, [-1])
  losses = losses * mask
  loss = tf.reduce_sum(losses) / tf.reduce_sum(mask)
  return loss

def train_load(FLAGS, bt):
  # batch normalisation apparently? https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  
    #with tf.device(FLAGS.device):
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph( FLAGS.save_path + "/" + FLAGS.save_name + '.meta')
      #saver.restore(sess, FLAGS.save_path + "/" + FLAGS.save_name) 
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_path))
      graph = sess.graph
      _train(FLAGS, graph, bt, sess, reload=True)

def train(FLAGS, graph, bt): 
  # batch normalisation apparently? https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
    with tf.device(FLAGS.device):
      with tf.Session(graph=graph) as sess:
        _train(FLAGS, graph, bt, sess)

def _train(FLAGS, graph, bt, sess, reload=False):
  ''' Run the training session once we have a graph, training methodology and a dataset.
  FLAGS is the NeuralGlobals class. Graph is a tensorflow graph object and 
  datasets is a tuple as received from gen_data.'''
  # Pull out the bits of the graph we need
  ginput = graph.get_tensor_by_name("train_input:0")
  gtest = graph.get_tensor_by_name("train_test:0")
  goutput = graph.get_tensor_by_name("output:0")
  gmask = graph.get_tensor_by_name("dmask:0")
  gprob = graph.get_tensor_by_name("keepprob:0")
  
   
  # Working out the accuracy
  basic_error = None
  glabel = None
  gpred = None

  if FLAGS.type_out == batcher.BatchTypeOut.CAT:
    glabel = graph.get_tensor_by_name("labels:0")
    gpred = graph.get_tensor_by_name("prediction:0")
    glogits = graph.get_tensor_by_name("logits:0")
    glength = graph.get_tensor_by_name("length:0")
    # This seems not to move and tends to produce NaNs
    #basic_error = cost_logit(glogits, glabel, glength, FLAGS)
    basic_error = cost_cat(gpred, gtest, glength, FLAGS)
  else:
    basic_error = cost(goutput, gtest, FLAGS)

  # https://stackoverflow.com/questions/36706379/how-to-exactly-add-l1-regularisation-to-tensorflow-error-function#42580575
  # TODO - this needs to be a bit better used
  l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
  weights = tf.trainable_variables() # all vars of your graph
  print(weights)
  #ll = [greg0, greg1, greg2, greg3]
  #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [gweights, gweights2, gweights3])
  #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, ll)
  regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
  regularized_loss = basic_error + regularization_penalty # this loss needs to be minimized


  # Setup all the logging for tensorboard 
  esum = tf.summary.scalar("TrainingError",basic_error)
  vsum = tf.summary.scalar("ValidationError", basic_error)
  train_writer = tf.summary.FileWriter(FLAGS.save_path + '/summaries/train',graph)
  merged = tf.summary.merge_all() 
  # https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow   
  train_step = None

  if not reload:

    # There is a bug here. If we don't add optimizer.name this won't serialise
    optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate, name="BUG")
    optimizer.name = "BUG"

    #train_step = optimizer.minimize(regularized_loss)
    #train_step = optimizer.minimize(basic_error, global_step=global_step)
    train_step = optimizer.minimize(basic_error)

    # Gradient clipping - stops things from exploding
    #gvs = optimizer.compute_gradients(basic_error)
    #gvs = optimizer.compute_gradients(regularized_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #train_step = optimizer.apply_gradients(capped_gvs) 
   
    tf.global_variables_initializer().run()
    tf.add_to_collection("optimizer", optimizer)
  else:
    train_step = tf.get_collection("optimizer")[0]    

  print('Initialized')	

  # The actual running
  total_steps = 0 
  error_history = []
  eval_limit = 50

  for epoch in range(0,FLAGS.num_epochs):
    stepnum = 0
    bt.reset()

    while bt.has_next_batch_random(batcher.SetType.TRAIN):
      (batch_is, batch_os, loop_t) = bt.next_batch(batcher.SetType.TRAIN, randset=True )
      (batch_iv, batch_ov, loop_v) = bt.random_batch(batcher.SetType.VALIDATE)
  
      labels_t = bt.batch_to_labels(batch_os)
      labels_v = bt.batch_to_labels(batch_ov)
    

      # For some reason, if the batches are not ALL the same size, we get a crash
      # so I reject batches smaller than the one set
      # This seems to be due to the gradient clipping, whatever that is?
      if len(batch_is) != FLAGS.batch_size or len(batch_iv) != FLAGS.batch_size:
        continue

      mask = bt.create_mask(batch_is)
      feed_dict = {ginput: batch_is, gtest: batch_os, gmask: mask, gprob: FLAGS.dropout}
      if FLAGS.type_out == batcher.BatchTypeOut.CAT:
        feed_dict[glabel] = labels_t

      summary, _ = sess.run([merged, train_step], feed_dict= feed_dict)
     
      # Evaluate and print the error, based on the validation set
      # We also record the error and perform an early stop and save
      if stepnum % eval_limit == 0:
        mask = bt.create_mask(batch_iv) 
        feed_dict = {ginput: batch_iv, gtest: batch_ov, gmask: mask, gprob: 1.0}
        if FLAGS.type_out == batcher.BatchTypeOut.CAT:
          feed_dict[glabel] = labels_v
     
        validation_accuracy, validation_sum = sess.run([basic_error, vsum], feed_dict=feed_dict)   

        mask = bt.create_mask(batch_is)
        feed_dict = {ginput: batch_is, gtest: batch_os, gmask: mask, gprob: 1.0}
        if FLAGS.type_out == batcher.BatchTypeOut.CAT:
          feed_dict[glabel] = labels_t
     
        train_accuracy, train_sum = sess.run([basic_error, esum ], feed_dict=feed_dict)   

        train_writer.add_summary(validation_sum, total_steps)
        train_writer.add_summary(train_sum, total_steps)

        print('epoch %d, step %d, training accuracy %g, validation accuracy %g' % (epoch, stepnum, train_accuracy, validation_accuracy))
        
        eval_limit = max(1, 5 * int(math.floor(validation_accuracy / FLAGS.absolute_error)))
        
        error_history.append(validation_accuracy)
        if len(error_history) > FLAGS.error_window:
          error_history = error_history[1:]

      train_writer.add_summary(summary, total_steps)
      stepnum += 1
      total_steps += 1
      global_step = total_steps
    
      if stepnum % eval_limit == 0:
        # Run a quick test
        test.predict(FLAGS, sess, graph, bt) 

        # Test for early quit
        if len(error_history) == FLAGS.error_window:
          diff = 0
          for i in range(1, FLAGS.error_window):
            diff += math.fabs(error_history[i] - error_history[i-1])
          diff /= FLAGS.error_window
          print("Diff", diff, "Err:", error_history[-1])
          if diff <= FLAGS.error_delta and error_history[-1] < FLAGS.absolute_error:
            print("Low error reached. Saving at epoch:", epoch)
            saver = tf.train.Saver()
            saver.save(sess, FLAGS.save_path + "/" + str(epoch) + "_" + FLAGS.save_name)
            sys.exit()
    
    # Save a version each epoch
    saver = tf.train.Saver()
    saver.save(sess, FLAGS.save_path + "/" + FLAGS.save_name) 
