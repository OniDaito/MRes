"""
stats.py - Generate stats such as RMSD for our net
author : Benjamin Blundell
email : me@benjamin.computer

"""
import os, sys, signal, subprocess, math
from Bio.PDB import *
import numpy as np

amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO","SEC", "SER", "THR", "TYR", "TRP", "VAL"]
ABDB_TOTAL_COUNT = 33579
ADDB_RES_COUNT =[1786,1512,929,4315,112,956,328,4206,699,562,1361,426,1034,2286,1071,0,1791,1089,6438,927,1760]


def gen_pdb_pairs(path):
  pdb_files = []

  for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
      pdb_extentions = ["pdb","PDB"]
      if any(x in filename for x in pdb_extentions) and "all" not in filename and "alg" not in filename:
        pdb_files.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))

  pdb_files.sort()
  pdb_pairs = []
  
  # pair up for pred and real
  for i in range(0,len(pdb_files)-1,2):
    pdb_pairs.append((pdb_files[i], pdb_files[i+1]))

  return pdb_pairs

def gen_res_pairs(path):
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

def gen_rmsd(pairs):
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

    try:
      pro = subprocess.run(["pdbfit", "-c", pair[0], pair[1] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      tr = float(tr.replace("RMSD  ",""))
      rmsds.append((os.path.basename(pair[0]), os.path.basename(pair[1]), num_residues, tr))

    except:
      print("Failed on ", pair[0], pair[1])

  return rmsds

def _gen_end(path):
  parser = PDBParser()
  distances = []
  bn = os.path.basename(path)
  try:
    st = parser.get_structure(bn, path)  
  except:
    print("Failed to load",path)
    return -1

  models = st.get_models() # should only be one
  num_residues = 0
  for model in models:
    atoms = []
    for atom in model.get_atoms():
      atoms.append(atom)

    # Should just be CAs
    x0 = atoms[0].get_coord()[0]
    y0 = atoms[0].get_coord()[1]
    z0 = atoms[0].get_coord()[2]
        
    x1 = atoms[-1].get_coord()[0]
    y1 = atoms[-1].get_coord()[1]
    z1 = atoms[-1].get_coord()[2]

    dist = math.sqrt( (x0-x1) * (x0-x1) + (y0-y1) * (y0-y1) + (z0-z1) * (z0-z1))
    distances.append(dist)
  return distances

def gen_end(pairs):
  dp = []
  dr = []
  for pair in pairs:
    # 0 is predicted, 1 is real
    d_pred = _gen_end(pair[0])
    d_real = _gen_end(pair[1])
    if d_pred == -1 or d_real == -1:
      continue
    dp.append(d_pred[0])
    dr.append(d_real[0]) # should only be one

  distances = list(zip(dp,dr))

  # Lets return some stats here
  mn = 7.537065927577005
  sd = 1.13033939846 * 2.0
  
  valid_r = 0
  valid_p = 0
  both_valid = 0
  both_wrong = 0
  real_right = 0
  pred_right = 0
  
  new_pairs = []

  idx = 0
  for dd in distances:
    vp = False
    vr = False
    if not(dd[0] > mn + sd or dd[0] < mn - sd):
      valid_p += 1
      new_pairs.append(pairs[idx])
      vp = True

    if not(dd[1] > mn + sd or dd[1] < mn - sd):
      valid_r += 1
      vr = True
    #else:
    #  print(pairs[idx])

    if vr and vp:
      both_valid += 1
    elif not vr and not vp:
      both_wrong += 1
    elif vr and not vp:
      real_right += 1
    elif not vr and vp:
      pred_right += 1
    idx += 1

  total = len(distances)

  # Bit nasty!
  return (valid_r, valid_p, total, new_pairs)

def pair_to_res(pair):
  residues = []
  with open(pair[0],'r') as f:
      for line in f.readlines():
        line = line.replace(",","")
        line = line.replace(":","")
        tokens = line.split(" ")
         # Convert to sin and cos of each
        phi = float(tokens[1])
        psi = float(tokens[2])
        residues.append([tokens[0],math.cos(phi),math.sin(phi),math.cos(psi),math.sin(psi)])
    
  idx = 0
  with open(pair[1],'r') as f:
    for line in f.readlines():
      line = line.replace(",","")
      line = line.replace(":","")
      tokens = line.split(" ")
   
      # Convert to sin and cos of each
      phi = float(tokens[1])
      psi = float(tokens[2])

      residues[idx].append(math.cos(phi))
      residues[idx].append(math.sin(phi))
      residues[idx].append(math.cos(psi))
      residues[idx].append(math.sin(psi))
    
      idx +=1

  return residues

def pair_to_diff(pair):
  residues = []
  
  with open(pair[0],'r') as f:
      for line in f.readlines():
        line = line.replace(",","")
        line = line.replace(":","")
        tokens = line.split(" ")
         # Convert to sin and cos of each
        phi = float(tokens[1])
        psi = float(tokens[2])
        residues.append([tokens[0],phi,psi])
    
  idx = 0
  with open(pair[1],'r') as f:
    for line in f.readlines():
      line = line.replace(",","")
      line = line.replace(":","")
      tokens = line.split(" ")
    
      # Convert to sin and cos of each
      phi = float(tokens[1])
      psi = float(tokens[2])
  
      residues[idx].append(phi)
      residues[idx].append(psi)
    
      idx +=1

  return residues

def gen_res_errors(pairs):
  res_errors = {}

  for aa in amino_acid_bitmask:
    res_errors[aa] = []

  for pair in pairs:
    residues = pair_to_res(pair)

    for res in residues:
      res_errors[res[0]].append(math.pow(res[1] - res[5],2))
      res_errors[res[0]].append(math.pow(res[2] - res[6],2))
      res_errors[res[0]].append(math.pow(res[3] - res[7],2))
      res_errors[res[0]].append(math.pow(res[4] - res[8],2))

  return res_errors

def gen_diff_errors(pairs):
  res_errors = {}

  avg_phi = 0
  avg_psi = 0
  cnt = 0
  rl_phi = []
  rl_psi = []

  for aa in amino_acid_bitmask:
    res_errors[aa] = []

  for pair in pairs:
    residues = pair_to_diff(pair)

    for res in residues:
    
      phi0 = res[1] 
      phi1 = res[3]
      psi0 = res[2]
      psi1 = res[4]
    
      rl_phi.append(phi1)
      rl_psi.append(psi1)

      phi0 = (math.degrees(phi0) + 180) % 360
      phi1 = (math.degrees(phi1) + 180) % 360
      psi0 = (math.degrees(psi0) + 180) % 360
      psi1 = (math.degrees(psi1) + 180) % 360

      diff_phi = min(phi1-phi0, 360 + phi0 - phi1)
      if phi1 < phi0:
        diff_phi = min(phi0-phi1, 360 + phi1 - phi0)
      
      diff_psi = min(psi1-psi0, 360 + psi0 - psi1)
      if psi1 < psi0:
        diff_psi = min(psi0-psi1, 360 + psi1 - psi0)

      avg_phi += diff_phi
      avg_psi += diff_psi
      cnt += 1
      
      res_errors[res[0]].append(diff_phi)
      res_errors[res[0]].append(diff_psi)

  sd_phi = np.std(rl_phi)
  sd_psi = np.std(rl_psi)

  return (res_errors, avg_phi / cnt, avg_psi / cnt, sd_phi, sd_psi)

def gen_res_avg(res_errors, pairs):
  for aa in amino_acid_bitmask:
    # Double divide by length here but still
    mean_err = (sum(res_errors[aa]) / len(res_errors)) / len(pairs)
    print (aa,mean_err)

def gen_worst(rmsds, pairs):
  ''' Find out what is making the worst offenders the worst.'''
  worst = {}
  final_rmsds = {}
  worst_acids = {}
  num_worst = {}
  for acid in amino_acid_bitmask:
    worst_acids[acid] = 0

  for rmsd in rmsds:
    key = os.path.basename(rmsd[0])
    key = key.split("_")[0]
    worst[key] = []
    final_rmsds[key] = rmsd[3]
    num_worst[key] = 0

  for pair in pairs:
    residues = pair_to_diff(pair)
    key = os.path.basename(pair[0]).split("_")[0]
    idx = 0
    res_errors = []
    for res in residues:
    
      phi0 = res[1] 
      phi1 = res[3]
      psi0 = res[2]
      psi1 = res[4]

      phi0 = (math.degrees(phi0) + 180) % 360
      phi1 = (math.degrees(phi1) + 180) % 360
      psi0 = (math.degrees(psi0) + 180) % 360
      psi1 = (math.degrees(psi1) + 180) % 360

      diff_phi = min(phi1-phi0, 360 + phi0 - phi1)
      if phi1 < phi0:
        diff_phi = min(phi0-phi1, 360 + phi1 - phi0)
      
      diff_psi = min(psi1-psi0, 360 + psi0 - psi1)
      if psi1 < psi0:
        diff_psi = min(psi0-psi1, 360 + psi1 - psi0)

      res_errors.append((idx,diff_phi,diff_psi,res[0]))

      idx += 1
    
    worst[key] = res_errors

  worst_keys = []
  for key in worst.keys():
    try:
      if final_rmsds[key] > 2.0:
        worst_keys.append(key)
    except:
      pass

  ANGLE = 120
  for key in worst_keys:
    for res in worst[key]:
      if res[0] >= ANGLE or res[1] >= ANGLE:
        worst_acids[res[3]] += 1
        num_worst[key] += 1

  for key in worst_keys: 
    print(key, final_rmsds[key], num_worst[key], len(worst[key]), float(num_worst[key]) / float(len(worst[key])) * 100.0)

  print(worst_acids)
  return (worst_keys, worst,final_rmsds)

def plot_rmsd_errors(errors):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import stats

  x = []
  y = []
  for e in errors:
    x.append(e[2])
    y.append(e[3])

  #n, bins, patches = plt.hist(dist_errors, 50, facecolor='g', alpha=0.75)

  plt.scatter(x,y,alpha=0.1)
  #plt.plot(x,y)

  plt.xlabel('Loop length / # of residues')
  plt.ylabel('RMSD Error (Angstroms)')
  plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
  print(stats.spearmanr(x,y))

def plot_res_errors(errors):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import stats

  for label in amino_acid_bitmask:
    n, bins, patches = plt.hist(errors[label], 18, facecolor='g', alpha=0.75, rwidth=0.5)
    plt.title('Residue error - ' + label + " total: " + str(len(errors[label])))
    plt.xlabel('Degrees of error (in 10 degree bins)')
    plt.ylabel('#') 
    plt.show()

  #plt.scatter(x,y,alpha=0.1)
  #plt.plot(x,y)

  #plt.xlabel('Loop length / # of residues')
  #plt.ylabel('RMSD Error (Angstroms)')
  #plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  #plt.grid(True)
  #plt.show()
  #print(stats.spearmanr(x,y))


def do_stats(path):
  pairs = gen_pdb_pairs(path)
  endpoints = gen_end(pairs)
  print("Real Endpoint Correct: ", endpoints[0] / endpoints[2] * 100.0)
  print("Pred Endpoint Correct: ", endpoints[1] / endpoints[2] * 100.0)
  new_pairs = endpoints[3]
  #rmsds = gen_rmsd(new_pairs)
  rmsds = gen_rmsd(pairs)
  pairs = gen_res_pairs(path)
  (reserrors, avg_phi, avg_psi, sd_phi, sd_psi) = gen_diff_errors(pairs)
  gen_res_avg(reserrors, pairs)
  #print(reserrors)
  #plot_res_errors(reserrors)
  #plot_rmsd_errors(rmsds) 
 
  gen_worst(rmsds, pairs)

  min_rmsd = 100
  max_rmsd = 0
  mean = 0
  trmsds = []
  
  for rmsd in rmsds:
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
  print ("Size:", len(pairs), len(new_pairs))
  print ("Avg Phi/Psi Diff/RealSD:", avg_phi, avg_psi, sd_phi, sd_psi)
  
  import csv
  with open('rmsds.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["predicted file", "real file", "number of residues", "rmsd"])
    for rmsd in rmsds:
      writer.writerow(rmsd)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Generate statistics for a particular Neural Net.')
  parser.add_argument('nnpath', metavar='D', type=str, help='path to neural net directory')

  args = vars(parser.parse_args())
  if args['nnpath']:
    do_stats(args['nnpath'])

