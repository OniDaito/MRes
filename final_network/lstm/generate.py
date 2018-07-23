"""
generate.py - generate all the PDBs from the test set
author : Benjamin Blundell
email : me@benjamin.computer
"""

import os, sys, math
import tensorflow as tf
import numpy as np

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.util_neural import *
  from common import acids
  from common.pdb import loop_to_pdb
  from common.endpoint import InverseK
  from common import batcher
  from common.repping import Loop, Residue

def generate(FLAGS, bt, end_refine, start=-1, end=-1, train_set=False, abdb_only=False):
  ''' With FLAGS and a batcher, generate PDBs for the test set. Assumes a saved
  graph already exists. Tensor flow has a problem with it's memory it seems
  so I've added a batch option to get around this.'''
  # Rather than a session withn a session, we save in memory and then run each one again
  real_models = []
  test_models = []
  loop_models = []

  #os.environ['CUDA_VISIBLE_DEVICES'] = ''
  #config = tf.ConfigProto(
  #    device_count = {'GPU': 0}
  #)
  
  genset = bt.test_set
  settype = batcher.SetType.TEST
  if train_set:
    genset = bt.train_set
    settype = batcher.SetType.TRAIN

  if start <= 0:
    start = 0
  if start + end >= len(genset):
    print("Start and end greater than test set size. Aborting.")
    return

  if end == -1:
    end = int(math.floor((len(genset) - 1) / bt.batch_size)) * bt.batch_size

  #with tf.Session(config=config) as sess:
  with tf.Session() as sess:
    graph = sess.graph
    saver = tf.train.import_meta_graph(FLAGS.save_path + FLAGS.save_name + '.meta')
    saver.restore(sess, FLAGS.save_path + FLAGS.save_name )
    # Get the tensors for our graph
    goutput = graph.get_tensor_by_name("output:0")
    ginput = graph.get_tensor_by_name("train_input:0")
    gmask = graph.get_tensor_by_name("dmask:0")
    gprob = graph.get_tensor_by_name("keepprob:0")  
    count = 0
    bt.offset(start, settype) # Start with an offset if one is passed
    
    with open(FLAGS.save_path + "/model_lookup.txt","w") as f:
      
      while bt.has_next_batch(settype) and count < (end - start):
        (batch_in, batch_out, loops) = bt.next_batch(settype)
        mask = bt.create_mask(batch_in)
        res = sess.run([goutput], feed_dict={ginput: batch_in, gmask: mask, gprob: 1.0 })
        
        for j in range(0, FLAGS.batch_size):
          real_residues = []
          test_residues = []
        
          # Collect residues together removing anything masked out
          for i in range(0,len(batch_in[j])):  
            res_name = bt.input_to_acid(batch_in[j][i])
            if res_name == None:
              break

            phi = 0 
            psi = 0

            if FLAGS.type_out == batcher.BatchTypeOut.SINCOS:
              phi = math.atan2(batch_out[j][i][0], batch_out[j][i][1])
              psi = math.atan2(batch_out[j][i][2], batch_out[j][i][3]) 
            
            else: 
              ( phi, psi )= batcher.cat_to_angles(batch_out[j][i])

            real_residues.append(Residue(res_name, i, i, phi, psi, math.pi))

            if FLAGS.type_out == batcher.BatchTypeOut.SINCOS:
              phi = math.atan2(res[0][j][i][0], res[0][j][i][1])
              psi = math.atan2(res[0][j][i][2], res[0][j][i][3])
            else: 
              ( phi, psi )= batcher.cat_to_angles(res[0][j][i])

            test_residues.append(Residue(res_name, i, i, phi, psi, math.pi))

      
          # Write angles to files
          with open(FLAGS.save_path + "/" + str(count+start).zfill(4) + "_real.txt","w") as rw:
            for rest in real_residues:
              rw.write(acids.amino_to_label(rest._name) + "," + str(rest.phid()) + "," + str(rest.psid()) + "\n")  
          with open(FLAGS.save_path + "/" + str(count+start).zfill(4) + "_pred.txt","w") as pw: 
            for rest in test_residues:
              pw.write(acids.amino_to_label(rest._name) + "," + str(rest.phid()) + "," + str(rest.psid()) + "\n")
          
          real_models.append(real_residues)
          test_models.append(test_residues)
        
          loop_models.append(loops[j])
          f.write( str(count+start).zfill(4) + "," + str(loops[j]) + "\n")
          count += 1

  print("Generating", count, "models from", start, "to", start+count)

  from common.nerf import NeRF
  nerf_herder = NeRF()

  # Now we have real and test
  for i in range(0, count):  
    # Inverse kinematics refine endpoints section
    test_residues = test_models[i]
    real_residues = real_models[i]
    db_loop = loop_models[i]

    if end_refine:
      ik = InverseK(len(test_residues),lrate=0.4, steps=100)
      ik.train(test_residues, db_loop._endpoint)
 
    real_loop = Loop(str(i+start).zfill(4)) 
    for residue in real_residues:
      real_loop.add_residue(residue)

    test_loop = Loop(str(i+start).zfill(4))
    for residue in test_residues:
      test_loop.add_residue(residue)

    nerf_herder.gen_loop_coords(real_loop)
    loop_to_pdb(FLAGS.save_path + "/" + str(i+start).zfill(4) + "_real.pdb", real_loop)

    nerf_herder.gen_loop_coords(test_loop)
    loop_to_pdb(FLAGS.save_path + "/" + str(i+start).zfill(4) + "_pred.pdb", test_loop) 
