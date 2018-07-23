"""
stats_plot.py - Generate graphs and plots
author : Benjamin Blundell
email : me@benjamin.computer

"""

import os, sys, signal, subprocess, math
from Bio.PDB import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_errors(res_errors):
  import matplotlib.pyplot as plt

  n, bins, patches = plt.hist(res_errors["GLY"], 50, facecolor='g', alpha=0.75)
  plt.xlabel('Difference in degrees')
  plt.ylabel('Total')
  plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()

  n, bins, patches = plt.hist(res_errors["TYR"], 50, facecolor='g', alpha=0.75)
  plt.xlabel('Difference in degrees')
  plt.ylabel('Total')
  plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()

  n, bins, patches = plt.hist(res_errors["ASP"], 50, facecolor='g', alpha=0.75)
  plt.xlabel('Difference in degrees')
  plt.ylabel('Total')
  plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()

def plot_rmsd(rmsds):
  ''' Given rmsds in the format (p0, p1, cdr_len, rmsd, realname)
  plot histograms and charts of the distribution, and give a 
  correlation score.'''
  import matplotlib.pyplot as plt
  from scipy import stats

  # Plot of RMSD error versus length
  x = []
  y = []
  ld = {}
  min_rmsd = 100
  max_rmsd = 0 
  for e in rmsds:
    x.append(e[2])
    y.append(e[3])
    if not e[2] in ld.keys():
      ld[e[2]] = []
    ld[e[2]].append(e[3])

    if e[3] > max_rmsd:
      max_rmsd = e[3]
    if e[3] < min_rmsd:
      min_rmsd = e[3]

  tl = list(ld.keys())
  tl.sort()
  min_length = tl[0]
  max_length = tl[-1]
  
  plt.scatter(x,y,alpha=0.1)
  #plt.plot(x,y)

  plt.xlabel('Loop length / # of residues')
  plt.ylabel('RMSD Error (Angstroms)')
  plt.title('Residue error')
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
  print(stats.spearmanr(x,y))

  # Violin plot of reconstruct
  errors = []
  for i in range(0,33):
    errors.append([0])

  for i in range(len(x)):
    errors[x[i]].append(y[i]) 

  fig, ax = plt.subplots()
  ax.violinplot(errors, range(0,33), points=20, widths=0.3, showmeans=True, showextrema=True, showmedians=True)  
  plt.xticks(range(len(errors)), range(0,33))
  plt.xlabel('Loop length')
  plt.ylabel('RMSD Ca in Angstroms')
  fig.suptitle("Violin Plot of RMSD Ca between Real and Predicted Loops")
  plt.show()

  nbins = int(math.floor(max_rmsd * 2))
  hmax = 0
  # Histogram of the total RMSD errors bucketed
  n, bins, patches = plt.hist(y, nbins, facecolor='b', alpha=0.75, range=(0,max_rmsd))

  for b in n:
    if b > hmax:
      hmax = int(math.floor(b))
  
  hmax +=10 
  plt.ylabel('frequency')
  plt.xlabel('RMSD Carbon Alpha Error - 0.5 Angstrom bin')
  plt.title('RMSD Carbon Alpha Complete Validation Set')
  plt.ylim(0,hmax)
  plt.show()

  # Histograms per CDR length
  for length in tl:
    n, bins, patches = plt.hist(ld[length], nbins, facecolor='b', alpha=0.75, range=(0,max_rmsd))
    plt.ylabel('frequency')
    plt.xlabel('RMSD Carbon Alpha Error - 0.5 Angstrom bin')
    plt.title('RMSD Carbon Alpha Validation Set - CDR Length ' + str(length))
    plt.ylim(0,hmax)
    plt.show()
 
  # https://matplotlib.org/examples/mplot3d/hist3d_demo.html
  # 3D hist
  hist, xedges, yedges = np.histogram2d(x, y, bins=len(tl), range=[[min_length, max_length ], [min_rmsd, max_rmsd]])
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
  xpos = xpos.flatten('F')
  ypos = ypos.flatten('F')
  zpos = np.zeros_like(xpos)

  plt.xlabel('Loop length')
  plt.ylabel('RMSD error (angstroms)')

  # Construct arrays with the dimensions for the 16 bars.
  dx = 0.5 * np.ones_like(zpos)
  dy = dx.copy()
  dz = hist.flatten()
  ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
  plt.show()


