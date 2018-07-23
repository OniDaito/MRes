"""
stats_real.py - Generate RMSD but on the real data
author : Benjamin Blundell
email : me@benjamin.computer

"""

import os, sys, signal, subprocess, math
from Bio.PDB import *
import numpy as np

amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO","SEC", "SER", "THR", "TYR", "TRP", "VAL"]

def gen_pdb_pairs(path):
  ''' Generate pairs of pdb files from the path.'''
  pdb_files = []

  for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
      pdb_extentions = ["pdb","PDB"]
      if any(x in filename for x in pdb_extentions) and "algn" not in filename:
        pdb_files.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  pdb_files.sort()
  pdb_pairs = []
  
  # pair up for pred and real
  for i in range(0,len(pdb_files)-1,2):
    pdb_pairs.append((pdb_files[i], pdb_files[i+1]))

  return pdb_pairs

def gen_angle_pairs(path):
  ''' Generate pairs of text files containing the angles from the path.'''
  res_files = []

  for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
      res_extentions = ["txt","TXT"]
      if any(x in filename for x in res_extentions):
        res_files.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  res_files.sort()
  res_pairs = []
  
  # pair up for pred and real
  for i in range(0,len(res_files)-1,2):
    res_pairs.append((res_files[i], res_files[i+1]))

  return res_pairs

def gen_rmsd_loop(l0, l1):
  ''' Given two loops, create two pdb files and compute the RMSD.'''
  from common import pdb
  pdb.loop_to_pdb("./l0.pdb", l0)
  pdb.loop_to_pdb("./l1.pdb", l1)
  
  try:
    pro = subprocess.run(["pdbfit", "-c", "./l0.pdb", "./l1.pdb" ], stdout=subprocess.PIPE)
    tr = pro.stdout.decode()
    tr = float(tr.replace("RMSD  ",""))
    return tr
  except:
    print("Failed on ", l0)

  return -1

def gen_rmsd_scaff(l0, l1):
  ''' Given two loops, create two pdb files and compute the RMSD but don't do any fitting.'''
  from common import pdb
  pdb.loop_to_pdb("./l0.pdb", l0)
  pdb.loop_to_pdb("./l1.pdb", l1)
  
  try:
    pro = subprocess.run(["pdbcalcrms", "-c", "./l0.pdb", "./l1.pdb" ], stdout=subprocess.PIPE)
    tr = pro.stdout.decode()
    tr = float(tr.replace("RMS deviation over CA atoms: ",""))
    return tr
  except:
    print("Failed on ", l0)

  return -1


def gen_rmsd(pairs, model_lookup):
  ''' Generate the RMSD scores on the carbon alpha with the program pdbfit.'''
  rmsds = []
  parser = PDBParser()

  for pair in pairs:
    bn = os.path.basename(pair[0])
    try:
      st = parser.get_structure(bn, pair[0])  
    except:
      print("Failed on ", pair[0], pair[1])
      continue

    models = st.get_models() # should only be one
    num_residues = 0
    for model in models:
      for res in model.get_residues():
        num_residues += 1
  
    tname = os.path.basename(pair[0]).split(".")[0]
    tname = tname.replace("_pred","")
    real_name = model_lookup[tname] 
 
    try:
      pro = subprocess.run(["pdbfit", "-c", pair[0], pair[1] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      tr = float(tr.replace("RMSD  ",""))
      rmsds.append((os.path.basename(pair[0]), os.path.basename(pair[1]), num_residues, tr, real_name))
      # Write fitted to pdb file
      final_path = pair[0]
      final_path = final_path.replace("pred","algn")
      pro = subprocess.run(["pdbfit", "-w", "-c", pair[0], pair[1]], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      with open(final_path,"w") as f:
        f.write(tr)

    except:
      print("Failed on ", pair[0], pair[1])
    
  return rmsds

def create_lookup(path):
  ''' Look for the file model_lookup.txt to link our random set 
  of loops to the real loop.'''
  lookup = {}
  with open(path + "/model_lookup.txt","r") as f:
    for line in f.readlines():
      line = line.replace("\n","")
      tokens = line.split(",")
      lookup[tokens[0]] = tokens[1]

  return lookup

def prepare(path):
  pairs = gen_pdb_pairs(path)
  lookup = create_lookup(path)
  angle_pairs = gen_angle_pairs(path)
  return(pairs, lookup, angle_pairs)

def stats_rmsd(pairs, lookup, path, write_csv=False, draw_graphs=True):
  # Now perform our RMSD check on the real models
  from db import Grabber
  from common import pdb
  
  rmsds = gen_rmsd(pairs, lookup)
  rmsds.sort(key=lambda tup: tup[3])  # sorts in place
 
  g = Grabber()
  min_rmsd = 100
  max_rmsd = 0
  mean = 0
  
  reals = []
  trmsds = []
  for rmsd in rmsds:
    real_loop = g.grab(rmsd[4])
    pred_loop = pdb.loop_from_pdb(path + "/" + rmsd[0])
    error = gen_rmsd_loop(real_loop, pred_loop)

    reals.append((rmsd[0], rmsd[4], rmsd[2], error))

  reals.sort(key=lambda tup: tup[3])  # sorts in place
  
  for rmsd in reals:
    mean += rmsd[3]
    print(rmsd)
    trmsds.append(rmsd[3])
    if rmsd[3] > max_rmsd:
      max_rmsd = rmsd[3]
    if rmsd[3] < min_rmsd:
      min_rmsd = rmsd[3]

  mean /= len(reals)
  print ("RMSD Mean Average:", mean)
  print ("RMSD Min:", min_rmsd)
  print ("RMSD Max:", max_rmsd)
  print ("RMSD StdDev:", np.std(np.array(trmsds)))
  print ("Size:", len(reals))

  if write_csv:
    import csv
    with open(path + '/rmsds_real.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["predicted file", "real file", "number of residues", "rmsd"])
      for rmsd in rmsds:
        writer.writerow(rmsd)

  if draw_graphs: 
    # Batch the RMSDs and draw histograms
    import stats_viz
    stats_viz.plot_rmsd(reals)


def stats_scaffold(pdb_path, pairs, lookup, path, start = 0, end = -1, write_csv=False, draw_graphs=True):
  # Run the scaffold 
  from db import Grabber
  from common import pdb  
  from common import acids
  import common.settings
  from run.sequence import sequence

  print("Set size:", len(lookup))

  FLAGS = common.settings.NeuralGlobals()
  FLAGS.kinematic = True
  FLAGS.scaffold = True
  FLAGS.save_path = path
  if FLAGS.save_path[-1] != "/":
    FLAGS.save_path += "/"

  min_rmsd = 100
  max_rmsd = 0
  mean = 0 
  g = Grabber()
  reals = []
  trmsds = []
  count = 0

  for pet in lookup.items():
    real_loop = g.grab(pet[1])
    FLAGS.scaffold_path = pdb_path + "/" + pet[1] + ".pdb"
    print("SCAFF", FLAGS.scaffold_path)

    try:
      pro = subprocess.run(["cp", FLAGS.scaffold_path, path + "/" + pet[1] + "_real.pdb" ], stdout=subprocess.PIPE)
    except:
      print("Failed copy of", FLAGS.scaffold_path)
 
    resseq = ""

    for res in real_loop.get_residues():
      resseq += acids.amino_to_letter(res._name)
  
    print(resseq)

    sequence(FLAGS, resseq, output_name = path + "/" + pet[1] + "_pred.pdb")
    pred_loop = pdb.loop_from_pdb(path + "/" + pet[1] + "_pred.pdb")
    error = gen_rmsd_loop(real_loop, pred_loop)
    error_fix = gen_rmsd_scaff(real_loop, pred_loop)
    reals.append((pet[0], pet[1], len(resseq), error_fix))
    count +=1

    if count + start > end and end != -1:
      break

  reals.sort(key=lambda tup: tup[3])  # sorts in place
  
  for rmsd in reals:
    mean += rmsd[3]
    print(rmsd)
    trmsds.append(rmsd[3])
    if rmsd[3] > max_rmsd:
      max_rmsd = rmsd[3]
    if rmsd[3] < min_rmsd:
      min_rmsd = rmsd[3]

  mean /= len(reals)
  print ("RMSD Mean Average:", mean)
  print ("RMSD Min:", min_rmsd)
  print ("RMSD Max:", max_rmsd)
  print ("RMSD StdDev:", np.std(np.array(trmsds)))
  print ("Size:", len(reals))

  if write_csv:
    import csv
    with open(path + '/rmsds_real.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["predicted file", "real file", "number of residues", "rmsd"])
      for rmsd in reals:
        writer.writerow(rmsd)

  if draw_graphs: 
    # Batch the RMSDs and draw histograms
    import stats_viz
    stats_viz.plot_rmsd(reals)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Generate statistics for a particular Neural Net.')
  parser.add_argument('nnpath', metavar='path', type=str, help='path to neural net directory')
  parser.add_argument('--rmsd', action='store_true', help='Find the RMSD')
  parser.add_argument('--scaff', action='store_true', help='Find the RMSD with the scaffold kinematic')
  parser.add_argument('--realdir', help='The path to the real pdb directory')
  parser.add_argument('--graphs', action='store_true', help='Draw resulting graphs')
  parser.add_argument('--start', type=int, help='Start position in test set')
  parser.add_argument('--end', type=int, help='End position in test set')
  parser.add_argument('--csv', action='store_true', help='End position in test set')

  args = vars(parser.parse_args())
  path = "."
  if args['nnpath']:
    path = args['nnpath']

  (pdb_pairs, lookup, angle_pairs) = prepare(path)

  if args["rmsd"] == True:
    stats_rmsd(pdb_pairs, lookup, path, draw_graphs=args["graphs"])
  if args["scaff"] == True:
    if args["realdir"]:

      start = 0
      end = -1
      if args["start"]:
        start = int(args["start"])
      if args["end"]:
        end = int(args["end"])

      stats_scaffold(args["realdir"], pdb_pairs, lookup, path, start = start, end = end, write_csv = args["csv"], draw_graphs=args["graphs"])

