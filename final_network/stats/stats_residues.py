"""
stats_residue.py - look at each residue and compare
author : Benjamin Blundell
email : me@benjamin.computer

"""

import os, sys, signal, subprocess, math
from Bio.PDB import *
import numpy as np
import matplotlib.pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)

from common.repping import *

amino_acid_bitmask = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO", "SER", "THR", "TYR", "TRP", "VAL"]

def residues(angle_pairs):
  from stats_diff import pair_to_diff
  loops = []

  # Distribution of the acids 
  distrib = {} 
  res_errors_phi = {}
  res_errors_psi = {}

  for acid in amino_acid_bitmask:
    distrib[acid] = 0
    res_errors_phi[acid] = []
    res_errors_psi[acid] = []

  for pair in angle_pairs:
    residues = pair_to_diff(pair)
    loops.append(residues)

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

      res_errors_phi[res[0]].append(diff_phi)
      res_errors_psi[res[0]].append(diff_psi)
      
  for loop in loops:
    for residue in loop:
      distrib[residue[0]] +=1

  #plt.bar(range(len(distrib)), distrib.values(), align='center')
  #plt.xticks(range(len(distrib)), distrib.keys())
  #plt.show()

  res_errors_phi_list = []
  res_errors_psi_list = []

  for key in res_errors_phi.keys():
    res_errors_phi_list.append(res_errors_phi[key])
    res_errors_psi_list.append(res_errors_psi[key])

  fig, ax = plt.subplots()
  ax.violinplot(res_errors_phi_list, range(0,20), points=20, widths=0.3, showmeans=True, showextrema=True, showmedians=True)  
  plt.xticks(range(len(distrib)), distrib.keys())
  plt.xlabel('Amino Acid')
  plt.ylabel('Error between real and predicted angle in degrees')
  fig.suptitle("Violin Plot of Phi Errors by Residue")
  plt.show()

  fig, ax = plt.subplots()
  ax.violinplot(res_errors_psi_list, range(0,20), points=20, widths=0.3, showmeans=True, showextrema=True, showmedians=True)  
  plt.xticks(range(len(distrib)), distrib.keys())
  plt.xlabel('Amino Acid')
  plt.ylabel('Error between real and predicted angle in degrees')
  fig.suptitle("Violin Plot of Psi Errors by Residue")
  plt.show()


