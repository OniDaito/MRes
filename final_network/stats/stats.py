"""
stats.py - Generate stats such as RMSD for our net
author : Benjamin Blundell
email : me@benjamin.computer

Intro to our stats program. Generate statistics on 
our neural net and the various results. This file
will generate the basic RMSD scores using the 
program 'pdbfit' which must exist on the local path.

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
      if any(x in filename for x in res_extentions) and ("real" in filename or "pred" in filename):
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

STDDEV = 1.13033939846
LIMIT = 7.537065927577005

def endpoint_check_coord(sc, ec):
  d = math.sqrt((ec[0] - sc[0]) * (ec[0] - sc[0]) + (ec[1] - sc[1]) * (ec[1] - sc[1]) + (ec[2] - sc[2]) * (ec[2] - sc[2]))

  if d > LIMIT - (STDDEV*3) and d < LIMIT + (STDDEV*3):
    return True
  
  return False

def endpoint_check_model(model):
  residues = list(model.get_residues());

  start = residues[0]['CA']
  end = residues[-1]['CA']

  sc = start.get_coord()
  ec = end.get_coord()

  return endpoint_check_coord(sc,ec)

def gen_rmsd(pairs, model_lookup, max_length=-1, endpoint = False):
  ''' Generate the RMSD scores on the carbon alpha with the program pdbfit.'''
  rmsds = []
  parser = PDBParser()

  for pair in pairs:
    bn = os.path.basename(pair[0])
    try:
      st = parser.get_structure(bn, pair[0])  
    except:
      #print("Failed on ", pair[0], pair[1])
      continue

    models = st.get_models() # should only be one
    num_residues = 0
    for model in models:
      for res in model.get_residues():
        num_residues += 1

    fail = False

    # Filter on max length
    if max_length != -1 and num_residues > max_length:
      fail = True
  
    # Filter on endpoint
    if endpoint:
      bn = os.path.basename(pair[1])
      try:
        st = parser.get_structure(bn, pair[1])
        models = st.get_models() # should only be one
        for model in models:
          if not endpoint_check_model(model):
            print(pair[1],"fails endpoint check")
            fail = True
      except Exception as e:
        print("ARG",e)
        fail = True

    if not fail:
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
        #print("Failed on ", pair[0], pair[1])
        pass
  return rmsds

def create_lookup(path):
  ''' Look for the file model_lookup.txt to link our random set 
  of loops to the real loop.'''
  lookup = {}
  import os.path
  if os.path.isfile(path + "/model_lookup.txt"):
    with open(path + "/model_lookup.txt","r") as f:
      for line in f.readlines():
        line = line.replace("\n","")
        tokens = line.split(",")
        lookup[tokens[0]] = tokens[1]
  elif os.path.isfile(path + "/model_mapping.txt"):
    with open(path + "/model_mapping.txt") as f:
      record = False
      for line in f.readlines():
        line = line.replace("\n","")

        if line == "***test set***":
          record = True
        if line == "***validate set***":
          record = False

        if record:
          tokens = line.split(",")
          if len(tokens) == 2:
            key = str(tokens[0]).zfill(3)
            lookup[key] = tokens[1]

  return lookup

def prepare(path):
  pairs = gen_pdb_pairs(path)
  lookup = create_lookup(path)
  angle_pairs = gen_angle_pairs(path)
  return(pairs, lookup, angle_pairs)

def stats_rmsd_inter(pairs, lookup, path, endpoint_filter=False, max_length=-1, write_csv=False, draw_graphs=True):
  print ("*** GENERATING AGAINST RECONSTRUCT ***")
  rmsds = gen_rmsd(pairs, lookup, endpoint=endpoint_filter, max_length=max_length)
  min_rmsd = 100
  max_rmsd = 0
  mean = 0
  trmsds = []
  
  rmsds.sort(key=lambda tup: tup[3])  # sorts in place
  
  for rmsd in rmsds:
    print(rmsd)
    mean += rmsd[3]
    trmsds.append(rmsd[3])
    if rmsd[3] > max_rmsd:
      max_rmsd = rmsd[3]
    if rmsd[3] < min_rmsd:
      min_rmsd = rmsd[3]

  mean /= len(rmsds)
  print ("RMSD Mean Average:", mean)
  print ("RMSD Min:", min_rmsd)
  print ("RMSD Max:", max_rmsd)
  print ("RMSD StdDev:", np.std(np.array(trmsds)))
  print ("Size:", len(pairs))
  print ("Actual Size:", len(rmsds))

  if write_csv:
    import csv
    with open('rmsds_reconstructed.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["predicted file", "real file", "number of residues", "rmsd" , "realloop"])
      for rmsd in rmsds:
        writer.writerow(rmsd)

  if draw_graphs: 
    # Batch the RMSDs and draw histograms
    import stats_viz
    stats_viz.plot_rmsd(rmsds)

  #sys.exit() # for now

def omega_check(loop):
  residues = loop.get_residues()
  residues.pop()
  for residue in residues:
    if residue.omegad() > -160 and residue.omegad() < 160:
      return False 
  return True

def stats_rmsd_db(pairs, lookup, path, endpoint_filter=True, omega_filter=False, max_length =-1, write_csv=False, draw_graphs=True, old_model=False):
  print ("*** GENERATING AGAINST DB ***")
  # Now perform our RMSD check on the real models
  from db import Grabber
  from common import pdb
 
  rmsds = gen_rmsd(pairs, lookup) # Ignore endpoint here. We check in the DB
  gp = Grabber()
  gm = Grabber(mongo=True)
  min_rmsd = 100
  max_rmsd = 0
  mean = 0

  rmsds.sort(key=lambda tup: tup[3])  # sorts in place
  
  endpointers = []
  reals = []
  trmsds = []
  for rmsd in rmsds:
    real_loop = None
    if len(rmsd[4]) > 7:
      real_loop = gm.grab(rmsd[4])
    else:
      real_loop = gp.grab(rmsd[4])
  
    if real_loop == None:
      #print("Error on", rmsd[4])
      continue

    bb = []
    for atom in real_loop._backbone:
      if old_model :
         if atom.kind == "CA":
          bb.append(atom)
      else:
        if atom.kind == "CA" or atom.kind == "N" or atom.kind == "C":
          bb.append(atom)

    real_loop._backbone = bb

    pred_loop = pdb.loop_from_pdb(path + "/" + rmsd[0])
    
    if max_length != -1 and len(real_loop.get_residues()) > max_length:
      print("Failed Maxlength", rmsd[4])
      continue

    if endpoint_filter and (not endpoint_check_coord(real_loop._backbone[0].tuple(), real_loop._backbone[-1].tuple()) or not endpoint_check_coord(pred_loop._backbone[0].tuple(), pred_loop._backbone[-1].tuple())):
      print("Failed Endpoint", rmsd[4])
      endpointers.append((path + "/" + rmsd[0].replace("pdb","txt"), path + "/" + rmsd[1].replace("pdb","txt")))
      continue

    if omega_filter and not omega_check(real_loop):
      print("Failed Omega", rmsd[4])
      continue

    # Now do the compare
    #try:

    error = gen_rmsd_loop(real_loop, pred_loop)

    reals.append((rmsd[0], rmsd[4], rmsd[2], error))
    #except Exception as e:
    #  print("Failed on", rmsd, e)

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
    with open('rmsds_real.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["predicted file", "real file", "number of residues", "rmsd"])
      for rmsd in rmsds:
        writer.writerow(rmsd)

  if draw_graphs: 
    # Batch the RMSDs and draw histograms
    import stats_viz
    stats_viz.plot_rmsd(reals)

  if endpoint_filter:
    from stats_diff import torsion_error 
    torsion_error(endpointers)
      

def stats_rmsd(pairs, lookup, path, endpoint_filter=True, omega_filter = False, max_length = -1, write_csv=False, draw_graphs=True, old_model=False):
  #stats_rmsd_inter(pairs, lookup, path, endpoint_filter, max_length, write_csv, draw_graphs)
  stats_rmsd_db(pairs, lookup, path, endpoint_filter, omega_filter, max_length, write_csv, draw_graphs, old_model)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Generate statistics for a particular Neural Net.')
  parser.add_argument('nnpath', metavar='path', type=str, help='path to neural net directory')
  parser.add_argument('--rmsd', action='store_true', help='generate pdbs from test set')
  parser.add_argument('--triple', action='store_true', help='Look at the triples and consider the error')
  parser.add_argument('--rama', action='store_true', help='Plot Phi vs Psi')
  parser.add_argument('--pos', action='store_true', help='Plot position of errors')
  parser.add_argument('--old', action='store_true', help='Looking at the older model')
  parser.add_argument('--graphs', action='store_true', help='generate pdbs from test set')
  parser.add_argument('--endpoint', action='store_true', help='Restrict on endpoint position')
  parser.add_argument('--omega', action='store_true', help='Restrict on bad omega')
  parser.add_argument('--torsion', action='store_true', help='Results in torsion space')
  parser.add_argument('--residues', action='store_true', help='Stats on the residues')
  parser.add_argument('--maxlength', type=int, help='Restrict on loop length')

  args = vars(parser.parse_args())
  path = "."
  if args['nnpath']:
    path = args['nnpath']

  (pdb_pairs, lookup, angle_pairs) = prepare(path)

  omega = False
  if args["omega"] == True:
    omega = True

  max_length = -1
  if args["maxlength"]:
    max_length = int(args["maxlength"])

  if args["rmsd"] == True:
    stats_rmsd(pdb_pairs, lookup, path, args["endpoint"], omega, max_length, False, args["graphs"], args["old"])
  
  if args["rama"] == True:
    import stats_rama
    stats_rama.plot_rama(angle_pairs)
  
  if args["residues"] == True:
    import stats_residues
    stats_residues.residues(angle_pairs)
  
  if args["torsion"] == True:
    import stats_diff
    stats_diff.torsion_error(angle_pairs)

  if args["pos"] == True:
    import stats_pos
    stats_pos.plot_pos(angle_pairs)

  if args["triple"] == True:
    import stats_triple
    stats_triple.triple(angle_pairs)
