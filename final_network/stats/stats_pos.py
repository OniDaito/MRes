import os, sys, signal, subprocess, math
from Bio.PDB import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_pos(angle_pairs):
  import matplotlib.pyplot as plt
  from stats_diff import pair_to_diff
  
  loops = []

  for pair in angle_pairs:
    residues = pair_to_diff(pair)
    loops.append(residues)

  xpsi = []
  ypsi = []

  xphi = []
  yphi = []

  for loop in loops:
    pos = 0
    ll = 1.0 / len(loop)
    for residue in loop:
      phip = residue[1]
      psip = residue[2]
      
      phir = residue[3]
      psir = residue[4]
   
      phir = (phir + 180) % 360
      phip = (phip + 180) % 360
      psir = (psir + 180) % 360
      psip = (psip + 180) % 360

      diff_phi = min(phir-phip, 360 + phip - phir)
      if phir < phip:
        diff_phi = min(phip-phir, 360 + phir - phip)
      
      diff_psi = min(psir-psip, 360 + psip - psir)
      if psir < psip:
        diff_psi = min(psip-psir, 360 + psir - psip)

      xphi.append(pos*ll)
      yphi.append(diff_phi)

      xpsi.append(pos*ll)
      ypsi.append(diff_psi)
      pos +=1

  plt.scatter(xphi,yphi,alpha=0.1)
  plt.xlabel('Position')
  plt.ylabel('Error')
  plt.title('Phi Error versus position')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
 
  plt.scatter(xpsi,ypsi,alpha=0.1)
  plt.xlabel('Position')
  plt.ylabel('Error')
  plt.title('Psi Error versus position')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()

