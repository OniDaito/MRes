"""
test.py - report the error on this net
author : Benjamin Blundell
email : me@benjamin.computer
"""

import os, sys, math, pickle
import numpy as np
import tensorflow as tf

# Import common items
if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.util_neural import *
  from common import acids
  from common import batcher

def adist(a,b):
  a = a + 180
  b = b + 180
  a = a - b
  a = (a + 180) % 360 - 180
  return a

def predict(FLAGS, sess, graph, bt):
    
  # Find the appropriate tensors we need to fill in
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gprob = graph.get_tensor_by_name("keepprob:0")
    gtest = graph.get_tensor_by_name("train_test:0")
  
    # Grab validation data
    (vbatch_in, vbatch_out, loop_v) = bt.random_batch(batcher.SetType.VALIDATE)
    mask = bt.create_mask(vbatch_in)
    
    # Run session
    res = None

    if FLAGS.type_out == batcher.BatchTypeOut.CAT:
      gpred = graph.get_tensor_by_name("prediction:0")
      res = sess.run([gpred], feed_dict={ginput: vbatch_in, gmask: mask, gprob: 1.0})
    else:
      res = sess.run([goutput], feed_dict={ginput: vbatch_in, gmask: mask, gprob: 1.0})
    
    # Now lets output a random example and see how close it is, as well as working out the 
    # the difference in mean values. Don't adjust the weights though
    import random
    r = random.randint(0, len(vbatch_in)-1)
    residues = loop_v[r]._residues[:]

    if FLAGS.type_out == batcher.BatchTypeOut.SINCOS:
      print("Actual                   Predicted")
      diff_psi = 0
      diff_phi = 0
      for i in range(0,len(loop_v[r]._residues)):
        
        # TODO data representation is now shared between acids and batcher :/
        if FLAGS.type_in == batcher.BatchTypeIn.FIVED: 
          sys.stdout.write(acids.amino_to_label(acids.vector_to_acid(vbatch_in[r][i])))
        else:
          sys.stdout.write(acids.amino_to_label(acids.bitmask_to_acid(vbatch_in[r][i])))
        
        phi0 = 0
        psi0 = 0
        phi0 = math.degrees(math.atan2(vbatch_out[r][i][0], vbatch_out[r][i][1]))
        psi0 = math.degrees(math.atan2(vbatch_out[r][i][2], vbatch_out[r][i][3]))
        
        sys.stdout.write(": " + "{0:<8}".format("{0:.3f}".format(phi0)) + " ")
        sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi0)) + " ")
        phi1 = 0
        psi1 = 0
        phi1 = math.degrees(math.atan2(res[0][r][i][0], res[0][r][i][1]))
        psi1 = math.degrees(math.atan2(res[0][r][i][2], res[0][r][i][3]))

        residues[i]._phi = phi1
        residues[i]._psi = psi1
        residues[i]._omega = math.pi
        sys.stdout.write(" | " + "{0:<8}".format("{0:.3f}".format(phi1)) + " ")
        sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi1)))
        diff_psi += math.fabs(adist(psi0,psi1))
        diff_phi += math.fabs(adist(phi0,phi1))

        print("")
    else:
      print("Actual                   Predicted")
      diff_psi = 0
      diff_phi = 0
      for i in range(0,len(loop_v[r]._residues)):
        
        # TODO data representation is now shared between acids and batcher :/
        if FLAGS.type_in == batcher.BatchTypeIn.FIVED: 
          sys.stdout.write(acids.amino_to_label(acids.vector_to_acid(vbatch_in[r][i])))
        else:
          sys.stdout.write(acids.amino_to_label(acids.bitmask_to_acid(vbatch_in[r][i])))
      
        
        (phi0, psi0 )= batcher.cat_to_angles(vbatch_out[r][i])
        phi0 = math.degrees(phi0)
        psi0 = math.degrees(psi0)

        sys.stdout.write(": " + "{0:<8}".format("{0:.3f}".format(phi0)) + " ")
        sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi0)) + " ")
        sys.stdout.write("{0:<8}".format("{0:.0f}".format(batcher.get_cat(vbatch_out[r][i]))) + " ")

        (phi1, psi1 )= batcher.cat_to_angles(res[0][r][i])
      
        phi1 = math.degrees(phi1)
        psi1 = math.degrees(psi1)

        residues[i]._phi = phi1
        residues[i]._psi = psi1
        residues[i]._omega = math.pi
        sys.stdout.write(" | " + "{0:<8}".format("{0:.3f}".format(phi1)) + " ")
        sys.stdout.write("{0:<8}".format("{0:.3f}".format(psi1))) 
        sys.stdout.write("{0:<8}".format("{0}".format(batcher.get_cat(res[0][r][i]))) + " ")
        diff_psi += math.fabs(adist(psi0,psi1))
        diff_phi += math.fabs(adist(phi0,phi1))

        print("")

    cnt = len(loop_v[r]._residues)
    print( "Diff in Phi/Psi", diff_phi / cnt, diff_psi / cnt)

def test(FLAGS, bt, end_refine):
  ''' Run the network on a random validation example to get a feel for 
  the error function. '''
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph(FLAGS.save_path + FLAGS.save_name + '.meta')
    saver.restore(sess, FLAGS.save_path + FLAGS.save_name)

    predict(FLAGS, sess, graph, bt)
