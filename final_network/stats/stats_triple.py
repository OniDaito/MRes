"""
stats_triple.py - look at each triplet residue and compare
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

def _split(line):
  line = line.replace("\n", "")
  tokens = line.split(",")
  return (tokens[0], float(tokens[1]), float(tokens[2]))

def _difa(a, b):
  ''' angles in degrees. We need the difference between them.'''
  v = a - b
  return math.fabs((v + 180) % 360 - 180)

def triple(pairs):

  triple_error_phi = {}
  triple_error_psi = {}
  triple_phi = {}
  triple_psi = {}
  triple_error_com = {}

  res_error = {}

  #for t in acids.TRIPLES: 
  #  res_error[t] = []

  for pair in pairs:
    real_lines = []
    pred_lines = []

    with open(pair[0],"r") as f:
      pred_lines = f.readlines()
    
    with open(pair[1],"r") as f:
      real_lines = f.readlines()
  
    for i in range(0,len(real_lines)-2):
      rs0 = _split(real_lines[i]) 
      ps0 = _split(pred_lines[i]) 
      rs1 = _split(real_lines[i+1]) 
      ps1 = _split(pred_lines[i+1]) 
      rs2 = _split(real_lines[i+2]) 
      ps2 = _split(pred_lines[i+2]) 
      
      key = rs0[0]+ "-" + rs1[0] + "-" + rs2[0]
   
      # Just the central one
      #d_phi = (_difa(rs0[1], ps0[1]) + _difa(rs1[1], ps1[1]) + _difa(rs2[1], ps2[1]))
      #d_psi = (_difa(rs0[2], ps0[2]) + _difa(rs1[2], ps1[2]) + _difa(rs2[2], ps2[2]))
      d_phi = _difa(rs1[1], ps1[1])
      d_psi = _difa(rs1[2], ps1[2])
      d_com = d_phi + d_psi

      if key not in triple_phi.keys():
        triple_phi[key] = []
        triple_psi[key] = []

      triple_phi[key].append(d_phi)
      triple_psi[key].append(d_psi)


  triple_phi_list = []
  x = []
  y = []
  idx = 0
  for key in triple_phi.keys():
    for err in triple_phi[key]:
      x.append(idx)
      y.append(err)
    idx+=1

  plt.scatter(x,y,alpha=0.1, s=1.0)
  #plt.plot(x,y)

  plt.xlabel('3-mer combination index')
  plt.ylabel('Combined Phi Angle Error of Triple')
  plt.title('3-mer categories & phi angle error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
 
  triple_psi_list = []
  x = []
  y = []
  idx = 0
  for key in triple_psi.keys():
    for err in triple_psi[key]:
      x.append(idx)
      y.append(err)
    idx+=1

  plt.scatter(x,y,alpha=0.1, s=1.0)
  #plt.plot(x,y)

  plt.xlabel('Triple index number')
  plt.ylabel('Combined Psi Angle Error of Triple')
  plt.title('3-mer categories & psi angle error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
 
  return


  import operator
  sorted_phi = sorted(triple_error_phi.items(), key=operator.itemgetter(1))
  #print(sorted_phi)
  sorted_psi = sorted(triple_error_psi.items(), key=operator.itemgetter(1))
  #print(sorted_psi)
  sorted_com = sorted(triple_error_com.items(), key=operator.itemgetter(1))
  print(sorted_com)

  sorted_res = sorted(res_error.items(), key=operator.itemgetter(1))
  print(sorted_res)

  print("Combos phi, psi: ", len(sorted_phi), len(sorted_psi))

  top_phi = sorted_phi[-50:]
  top_psi = sorted_psi[-50:]
 
  for p in top_psi:
    for h in top_phi:
      if p[0] == h[0]:
        print(p[0], p[1], h[1])

  

