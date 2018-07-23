"""
score.py - generate then look in the DB for the closest 5 matches
author : Benjamin Blundell
email : me@benjamin.computer
"""

import os, sys, math, subprocess
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
  from stats.db import Grabber
 
def _grab_db(FLAGS, gp, real_name, final_name):
  real_loop = None
  real_loop = gp.grab(real_name)

  if real_loop == None:
    print("Error on", real_name)
    return None

  bb = []
  for atom in real_loop._backbone:
    if atom.kind == "CA" or atom.kind == "N" or atom.kind == "C":
      bb.append(atom)

  real_loop._backbone = bb
  loop_to_pdb(FLAGS.save_path + "/" + final_name, real_loop)


def grab_db(FLAGS, bt):
  gp = Grabber()
  gm = Grabber(mongo=True)
  
  #loops = bt.get_loop_set(batcher.SetType.TRAIN) 
  #loops = loops + bt.get_loop_set(batcher.SetType.VALIDATE)
  # For now, just go with test because loopdb is too big?
  loops = bt.get_loop_set(batcher.SetType.TEST)

  fname = 0
  for loop in loops:
    if len(loop._name) < 8:
      _grab_db(FLAGS, gp, loop._name, str(fname).zfill(4) + "_real.pdb")
    else:
      _grab_db(FLAGS, gm, loop._name, str(fname).zfill(4) + "_real.pdb")
    fname+=1

def gen_scores(FLAGS):
  ''' Look at the final directory and create a list of real and preds, testing
  each one and seeing which gets closest.'''
  pdb_reals = []
  pdb_preds = []

  for dirname, dirnames, filenames in os.walk(FLAGS.save_path):
    for filename in filenames:
      pdb_extentions = ["pdb","PDB"]
      if any(x in filename for x in pdb_extentions) and "real" in filename:
        pdb_reals.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  for dirname, dirnames, filenames in os.walk(FLAGS.save_path):
    for filename in filenames:
      pdb_extentions = ["pdb","PDB"]
      if any(x in filename for x in pdb_extentions) and "pred" in filename:
        pdb_preds.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  correct = 0
  antibody_loop = 0
  total = 0
  for pred in pdb_preds:
    scores = []
    num_lines_1 = sum(1 for line in open(pred))

    for real in pdb_reals:
      try:
        num_lines_2 = sum(1 for line in open(real))
        if num_lines_2 == num_lines_1:
          pro = subprocess.run(["pdbfit", "-c", pred, real ], stdout=subprocess.PIPE)
          tr = pro.stdout.decode()
          tr = float(tr.replace("RMSD  ",""))
          scores.append((real, tr))
      except Exception as e:
        print (e)
        #print("Failed on ", pred, real)
    
    sort_scores = sorted(scores, key=lambda tup: tup[1])
    pred_name = pred.replace("_pred.pdb","")
    pred_name = os.path.basename(pred_name)
    print("***pred - " + pred_name + "***")
    for score in sort_scores[:5]:
      print(score)
      if pred_name in score[0]:
        correct +=1
        break

    position = 0
    for score in sort_scores:
      if pred_name in score[0]:
        print(score)
        break
      position += 1
    
    total +=1
    
    if len(sort_score[0][0]) <= 7:
      # ABDB so write
      antibody_loop += 1
    
    print("Total Correct", correct, "of", total, "position", position, "of", len(sort_scores), ". # antibody loops chosen", antibody_loop)

def load_lookup(FLAGS):
  lookup = {}
  with open(FLAGS.save_path + "/model_lookup.txt","r") as f:
    for line in f.readlines():
      tokens = line.replace("\n","").split(",")
      lookup[tokens[0]] = tokens[1]

  return lookup

def rmsd_check(FLAGS, pred_name, top_scores):
  ''' Run an RMSD check on the already existing PDB files - the top scoring ones
  from torsion space.'''
  
  # Compare against the real version, not the predicted for now
  pname = pred_name.replace("_pred.txt","_real.pdb")
  scores = []

  for top in top_scores:
    rname = top[0].replace("_real.txt","_real.pdb")
    try:
      pro = subprocess.run(["pdbfit", "-c", pname, rname ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      tr = float(tr.replace("RMSD  ",""))
      scores.append(tr)
    except Exception as e:
      print (e)

  return scores

def torsion_rmsd(pred_file, real_file):
  pred_angles = []
  real_angles = []

  with open(pred_file,'r') as f:
    for line in f.readlines():
      tokens = line.replace("\n","").split(",")
      pred_angles.append((float(tokens[1]), float(tokens[2])))
  
  with open(real_file,'r') as f:
    for line in f.readlines():
      tokens = line.replace("\n","").split(",")
      real_angles.append((float(tokens[1]), float(tokens[2])))

  rmsd = 0

  for i in range(0,len(pred_angles)):

    phi0 = pred_angles[i][0]
    psi0 = pred_angles[i][1]

    phi1 = real_angles[i][0]
    psi1 = real_angles[i][1]

    diff_phi = min(phi1-phi0, 360 + phi0 - phi1)
    if phi1 < phi0:
      diff_phi = min(phi0-phi1, 360 + phi1 - phi0)
        
    diff_psi = min(psi1-psi0, 360 + psi0 - psi1)
    if psi1 < psi0:
      diff_psi = min(psi0-psi1, 360 + psi1 - psi0)

    diff_phi *= diff_phi
    diff_psi *= diff_psi

    rmsd += diff_phi + diff_psi

  rmsd = rmsd / (len(pred_angles) * 2)
  return rmsd


def plot_scatter(scorepath):
  ''' Plot the scores by lenth '''

  score_by_length = []
  avg = 0 

  with open(scorepath, "r") as f:
    for line in f.readlines():
      tokens = line.replace("\n","").split(",")

      if float(tokens[0]) != 0:
        score_by_length.append((int(tokens[1]), float(tokens[0])))
        avg += float(tokens[0])

  print ("Average", avg / len(score_by_length))

  import matplotlib.pyplot as plt
  from scipy import stats

  bins = {}

  for score in score_by_length:
    if score[0] not in bins.keys():
      bins[score[0]] = []
    bins[score[0]].append(score[1])

  fbins = []
  keys = list(bins.keys())
  keys.sort()

  for k in keys:
    fbins.append(bins[k])

  try:
    x,y = map(list,zip(*score_by_length))
    fig = plt.figure()
    ax = plt.subplot(111)
    #ax.scatter(x,y,alpha=0.1)
    ax.violinplot(fbins, keys, points=20, widths=0.3, showmeans=True, showextrema=True, showmedians=True)

    #nbins = 32
    #hmax = 0
    # Histogram of the total RMSD errors bucketed
    #n, bins, patches = plt.hist(y, nbins, facecolor='b', alpha=0.75, range=(0,32))

    plt.xlabel('Loop length / # of residues')
    plt.ylabel('RMSD Error Ca (Angstroms)')
    plt.title('Predicted real loop error against actual loop, by loop length.')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
    fig.savefig('foo.png')
  except Exception as e:
    print("Error on save fig", e)
  print(stats.spearmanr(x,y))

def gen_scores_torsion(FLAGS, lookup):
  ''' Look at the final directory and the angles, figuring out the scores in 
  torsion space.'''
  txt_reals = []
  txt_preds = []
  txt_extentions = ["txt","TXT"]
  
  for dirname, dirnames, filenames in os.walk(FLAGS.save_path):
    for filename in filenames:
      if any(x in filename for x in txt_extentions) and "real" in filename:
        txt_reals.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  for dirname, dirnames, filenames in os.walk(FLAGS.save_path):
    for filename in filenames:
      if any(x in filename for x in txt_extentions) and "pred" in filename:
        txt_preds.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  correct = 0
  total = 0
  cartesian_mean = 0
  cartesian_total = 0
  score_by_length = []
  antibody_loop = 0
  antibody_match = 0

  with open("scores.txt", "w") as f:
    for pred in txt_preds:
      scores = []
      num_lines_1 = sum([1 for line in open(pred)])

      for real in txt_reals:
        try:
          num_lines_2 = sum([1 for line in open(real)])
          if num_lines_2 == num_lines_1:
            score = torsion_rmsd(pred, real)
            scores.append((real, score))
        except Exception as e:
          print (e)
          #print("Failed on ", pred, real)
      
      sort_scores = sorted(scores, key=lambda tup: tup[1])
      cartesian_scores = rmsd_check(FLAGS, pred, sort_scores[:5])
      pred_name = pred.replace("_pred.txt","")
      pred_name = os.path.basename(pred_name)
      real_name = lookup[pred_name]

      print("***pred - " + pred_name + "***")
      for score in scores[:5]:
        if pred_name in score[0]:
          correct +=1

      if len(cartesian_scores) > 0:
        cartesian_mean += cartesian_scores[0]
        f.write(str(cartesian_scores[0]) + "," + str(num_lines_1) + "\n")
        cartesian_total += 1
        score_by_length.append((num_lines_1, cartesian_scores[0]))

      position = 0
      for score in sort_scores:
        if pred_name in score[0]:
          #print(score)
          break
        position += 1
      
      if len(real_name) <= 7:
        antibody_loop += 1
        match_name = sort_scores[0][0]
        match_name = match_name.replace("_real.txt","")
        match_name = os.path.basename(match_name)
        match_name = lookup[match_name]
        print(real_name)
        print(match_name)
        if len(match_name) <= 7:
          antibody_match += 1

      total +=1
      print("Total Correct", correct, "of", total, "position", position, "of", len(sort_scores), "with", antibody_loop,"abdb loops / matched:", antibody_match)
  
  cartesian_mean /= cartesian_total
  print ("Total RMSD mean", cartesian_mean)
  plot_scatter(score_by_length)


def score(FLAGS, bt):
  ''' With FLAGS and a batcher, generate PDBs and find the closest match.'''
 
  #gen_scores(FLAGS)
  gen_scores_torsion(FLAGS)
  sys.exit()

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
    idx = 0

    with open(FLAGS.save_path + "/model_lookup.txt","w") as f:
      
      while bt.has_next_batch(settype):
        (batch_in, batch_out, loops) = bt.next_batch(settype)
        mask = bt.create_mask(batch_in)
        res = sess.run([goutput], feed_dict={ginput: batch_in, gmask: mask, gprob: 1.0 })
        
        for j in range(0, FLAGS.batch_size):
          test_residues = []
        
          for i in range(0,len(batch_in[j])):  
            res_name = bt.input_to_acid(batch_in[j][i])
            if res_name == None:
              break

            phi = 0 
            psi = 0

            if FLAGS.type_out == batcher.BatchTypeOut.SINCOS:
              phi = math.atan2(res[0][j][i][0], res[0][j][i][1])
              psi = math.atan2(res[0][j][i][2], res[0][j][i][3])
            else: 
              ( phi, psi )= batcher.cat_to_angles(res[0][j][i])

            test_residues.append(Residue(res_name, i, i, phi, psi, math.pi))
          
          test_models.append(test_residues)
        
          loop_models.append(loops[j])
          f.write( str(idx).zfill(4) + "," + str(loops[j]) + "\n")
          idx += 1
          count += 1

  # Generate our test models
  print("Generating", count, "models")

  from common.nerf import NeRF
  nerf_herder = NeRF()

  for i in range(0, count):  
    # Inverse kinematics refine endpoints section
    test_residues = test_models[i]
    db_loop = loop_models[i]
    test_loop = Loop(db_loop._name)

    for residue in test_residues:
      test_loop.add_residue(residue)

    nerf_herder.gen_loop_coords(test_loop)
    loop_to_pdb(FLAGS.save_path + "/" + db_loop._name + "_pred.pdb", test_loop)
 
  # Should only need grab_db once - it rebuild AbDb PDBs with just the loops
  # we need.
  grab_db(FLAGS, bt)
  gen_scores(FLAGS)


if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--savepath', help='Specify the path to save the net to. Default ./saved/')
  parser.add_argument('--savename', help='Specify a save name prefix. Default model.ckpt.')
  parser.add_argument('--maxlength', type=int, help='Set the maximum loop length.')
  parser.add_argument('--picklename', help='Set the pickle filename.')
  parser.add_argument('--typein', type=int, help='How is the data represented')
  parser.add_argument('--typeout', type=int, help='How is the data represented')
  parser.add_argument('--grabdb', action='store_true', help='Grab database')
  parser.add_argument('--graphfile', help='Graph the scores given a score file.')

  args = parser.parse_args()

  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
 
  # Various local imports
  import common.settings
  import common.batcher as batcher
  from stats.db import Grabber
  from common.pdb import loop_to_pdb

  FLAGS = common.settings.NeuralGlobals()

  if args.graphfile:
    plot_scatter(args.graphfile)  
    sys.exit()

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
 
  if args.grabdb:
    # regen the real loops from the actual atoms stored in the DB
    bt = batcher.Batcher(max_cdr_length = FLAGS.max_cdr_length, typein=FLAGS.type_in, typeout=FLAGS.type_out)
    print("Loading data from pickle:", FLAGS.save_path + FLAGS.pickle_filename)
    if os.path.exists(FLAGS.save_path + FLAGS.pickle_filename + "_test"):
      bt.create_sets_from_pickle(FLAGS.save_path + FLAGS.pickle_filename)
    else:
      bt.create_sets_from_pickle(FLAGS.save_path + FLAGS.pickle_filename, partial_pickle=False)
 
    grab_db(FLAGS, bt)

  lookup = load_lookup(FLAGS)
  gen_scores_torsion(FLAGS, lookup)


