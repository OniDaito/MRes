import math
import numpy as np

amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO","SEC", "SER", "THR", "TYR", "TRP", "VAL"]


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

def pair_to_diff(pair, old=False):
  residues = []

  with open(pair[0],'r') as f:
      for line in f.readlines():
        if old:
          line = line.replace(",","")
          line = line.replace(":","")
          tokens = line.split(" ")
        else:
          tokens = line.split(",")
        phi = float(tokens[1])
        psi = float(tokens[2])
        residues.append([tokens[0],phi,psi])
    
  idx = 0
  with open(pair[1],'r') as f:
    for line in f.readlines():
      if old:
        line = line.replace(",","")
        line = line.replace(":","")
        tokens = line.split(" ")
      else:
        tokens = line.split(",")
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

def gen_acid_errors(pairs):
  res_errors = {}

  for aa in amino_acid_bitmask:
    res_errors[aa] = []

  for pair in pairs:
    residues = pair_to_diff(pair)

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

      res_errors[res[0]].append(diff_phi)
      res_errors[res[0]].append(diff_psi)
      
      print(diff_phi)

  return res_errors

def gen_diff_errors(pairs):

  final_errors = []
  errors_by_loop = []
  by_length = {}

  for i in range(1,33):
    by_length[i] = []

  for pair in pairs:
    print("ppp",pair)
    residues = pair_to_diff(pair)
    
    position = 0
    total = len(residues)
    worst_pos = 0
    worst = 0

    if total < 8:
      continue

    loop_errs = []

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

      if diff_phi > worst:
        worst = diff_phi
        worst_pos = position

      if diff_psi > worst:
        worst = diff_psi
        worst_pos = position

      loop_errs.append(diff_phi)
      loop_errs.append(diff_psi)

      position += 1

    errors_by_loop.append(loop_errs)
    
    fe = float(worst_pos / total)
    by_length[total].append(fe)
    final_errors.append(fe)

  return (final_errors, by_length, errors_by_loop )


def gen_res_avg(res_errors, pairs):
  for aa in amino_acid_bitmask:
    # Double divide by length here but still
    mean_err = (sum(res_errors[aa]) / len(res_errors)) / len(pairs)
    print (aa,mean_err)


def torsion_error(pairs):
  (errors, by_len, by_loop) = gen_diff_errors(pairs)
  sd = np.std(errors, ddof=1)
  print ("Mean error position", sum(errors) / len(errors), "stddev", sd)
  
  for k in list(by_len.keys()):
    if len(by_len[k]) > 0:
      sd = np.std(by_len[k], ddof=1)
      print ("Mean error position for ", k, ":", sum(by_len[k]) / len(by_len[k]), "stddev", sd)

  minsd = 180
  maxsd = 0

  for loop in by_loop:
    sd = np.std(loop, ddof=1)
    md = np.median(loop)
    mn = np.mean(loop)
    if sd > maxsd:
      maxsd = sd
    if sd < minsd:
      minsd = sd

    print("Loop Err (mean, median, sd)", mn, md, sd, minsd, maxsd)

