"""
session.py - Run our graph through a session
author : Benjamin Blundell
email : me@benjamin.computer

"""
import os, sys, math
import tensorflow as tf
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)

if __name__ == "__main__":

  from common.util_neural import *
  from common import acids

  print("Starting run of net...")

  import argparse
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--generate', action='store_true', help='generate pdbs from test set')
  parser.add_argument('--nettype', type=int, help='network type: 0 - lstm, 1 - seq')
  parser.add_argument('--score', action='store_true', help='Run scoring method 1')
  parser.add_argument('--genabdb', action='store_true', help='generate pdbs from test abdb only')
  parser.add_argument('--gentrain', action='store_true', help='generate pdbs from train set')
  parser.add_argument('--test', action='store_true', help='Run the saved net on test data.')
  parser.add_argument('--sequence', help='Run the saved net on a provided sequnce of amino acids. Use the single letter amino acid labels.')
  parser.add_argument('--startgen', type=int, help='Starting point in the generation set.')
  parser.add_argument('--endgen', type=int, help='End point in the generation set')
  parser.add_argument('--savepath', help='Specify the path to save the net to. Default ./saved/')
  parser.add_argument('--savename', help='Specify a save name prefix. Default model.ckpt.')
  parser.add_argument('--mongo', action='store_true', help='Use the mongo database instead of postgres.')
  parser.add_argument('--removeomega', action='store_true', help='Remove models with bad omega.')
  parser.add_argument('--removeend', action='store_true', help='Remove models with bad endpoints.')
  parser.add_argument('--datalimit', type=int, help='Limit the total number of data items.')
  parser.add_argument('--maxlength', type=int, help='Set the maximum loop length.')
  parser.add_argument('--batchsize', type=int, help='Set the batch size.')
  parser.add_argument('--framework', help='Framework PDB for endpoints and grafting.')
  parser.add_argument('--learningrate', help='Set the learning rate.')
  parser.add_argument('--picklename', help='Set the pickle filename.')
  parser.add_argument('--typein', type=int, help='How is the data represented')
  parser.add_argument('--typeout', type=int, help='How is the data represented')

  args = parser.parse_args()

  import common.settings
  FLAGS = common.settings.NeuralGlobals()
  FLAGS.learning_rate = 0.004
  FLAGS.num_epochs = 2000 # number of loops around the training set
  FLAGS.batch_size = 5
  FLAGS.kinematic = args.kinematic
  FLAGS.scaffold = False
  FLAGS.lstm_size = 256
  FLAGS.error_window = 5
  FLAGS.window_size = 5
  FLAGS.dropout = 0.5
  FLAGS.absolute_error = 0.1
  FLAGS.error_delta = 0.001
  FLAGS.decay_learning = False

  if args.batchsize:
    FLAGS.batch_size = int(args.batchsize)

  if args.learningrate:
    FLAGS.learning_rate = float(args.learningrate)

  if args.picklename:
    FLAGS.pickle_filename = args.picklename

  if args.savepath:
    FLAGS.save_path = args.savepath
    if FLAGS.save_path[-1] != "/":
      FLAGS.save_path += "/"

  if args.savename:
    FLAGS.save_name = args.savename

  if args.maxlength:
    FLAGS.max_cdr_length = args.maxlength
  
  if args.sequence:
    from sequence import *

    if args.framework:
      FLAGS.scaffold = True
      FLAGS.scaffold_path = args.framework
 
    sequence(FLAGS,args.sequence)
    sys.exit()

  nettype = 0
  if args.nettype:
    nettype = int(args.nettype)

  # Network type
  if nettype == 0 :
    from lstm.graph import create_graph
    from lstm.train import * 
    from lstm.generate import *
    from lstm.test import *
  elif nettype == 1:
    from seq.graph import create_graph
    from seq.train import *
    from seq.generate import *
    from seq.test import *

  import score

  # AMA2 List for testing - do not train, validate or test on these.
  ama_list = ["4MA3_1", "4MA3_2","4KUZ_1", "4KQ3_1", "4KQ4_1", "4M6M_1", "4M6O_1", "4MAU_1", "4M7K_1", "4KMT_1", "4M61_1", "4M61_2", "4M43_1"]

  # Load data using batcher, with saved if it exists
  # Here we setup the batcher
  from common import batcher
  
  FLAGS.type_in = batcher.BatchTypeIn.BITFIELD
  if args.typein:
    FLAGS.type_in = list(batcher.BatchTypeIn)[int(args.typein)]

  FLAGS.type_out = batcher.BatchTypeOut.SINCOS
  if args.typeout:
    FLAGS.type_out = list(batcher.BatchTypeOut)[int(args.typeout)]

  b = batcher.Batcher(batch_size = FLAGS.batch_size, max_cdr_length = FLAGS.max_cdr_length, typein=FLAGS.type_in, typeout=FLAGS.type_out)
  
  if not os.path.exists(FLAGS.save_path + FLAGS.pickle_filename) and not os.path.exists(FLAGS.save_path + FLAGS.pickle_filename + "_test"):
    from common import gen_data
    
    print("Grabbing data from database...")
    g = gen_data.Grabber(remove_bad_endpoint = args.removeend, remove_odd_omega = args.removeomega, mongo=args.mongo)
    limit = -1
    if args.datalimit:
      limit = args.datalimit
    (loops, summary, blockers) = g.grab(limit = limit)
    print("Gen Data Summary", summary)
    b.create_sets_from_loops(loops, blockers = blockers, ignores=ama_list)
    b.pickle_it(FLAGS.save_path + FLAGS.pickle_filename)
  else:
    print("Loading data from pickle:", FLAGS.save_path + FLAGS.pickle_filename)
    if os.path.exists(FLAGS.save_path + FLAGS.pickle_filename + "_test"):
      b.create_sets_from_pickle(FLAGS.save_path + FLAGS.pickle_filename)
    else:
      b.create_sets_from_pickle(FLAGS.save_path + FLAGS.pickle_filename, partial_pickle=False)
    
  # Score instead
  if args.score:
    score.score(FLAGS, b)

  # parse args
  if args.generate or args.genabdb or args.gentrain:

    s = -1
    e = -1
    if args.startgen:
      s = int(args.startgen)
    if args.endgen:
      e = int(args.endgen)
    if args.genabdb:
      generate(FLAGS, b, False, s, e, abdb_only=True)
    elif args.gentrain:
      generate(FLAGS, b, False, s, e, train_set=True)
    else:
      generate(FLAGS, b, False, s, e)

  elif args.test:
    test(FLAGS, b, False)
  else:
   
    print(FLAGS.save_path + "/" +  FLAGS.save_name + '.meta')
    if os.path.exists( FLAGS.save_path + "/" +  FLAGS.save_name + '.meta'):
      print("Loading existing graph for further training.")
      train_load(FLAGS, b) 
    else:
      print("Creating new graph for training.")
      new_graph = create_graph(FLAGS)
      train(FLAGS, new_graph, b)
  
  print("Finished session.")
