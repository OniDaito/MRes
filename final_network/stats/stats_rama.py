import os, sys, signal, subprocess, math
from Bio.PDB import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def _plot(x,y,title): 
  import matplotlib.pyplot as plt
  plt.scatter(x,y,alpha=0.1)
  plt.xlabel('Phi')
  plt.ylabel('Psi')
  plt.title(title)
  #plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show()
 
def plot_rama(angle_pairs):
  from stats_diff import pair_to_diff
  loops = []

  for pair in angle_pairs:
    residues = pair_to_diff(pair)
    loops.append(residues)

  xr = []
  yr = []

  xp = []
  yp = []

  for loop in loops:
    for residue in loop:
      if residue[4] != 0 and residue[1] != 0 and residue[2]  != 0 and residue[3] != 0: 
        xp.append(residue[1])
        yp.append(residue[2])
        xr.append(residue[3])
        yr.append(residue[4])

  _plot(xp, yp, 'Ramachandran Predicted Plot')
  _plot(xr, yr, 'Ramachandran Real Plot')

if __name__ == "__main__":
  # A useful function to display a ramachandran plot of our abdb data
  import psycopg2

  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.repping import *

  conn = psycopg2.connect("dbname=pdb_martin user=postgres") 
  final_loops = []
  rr = 0
  
  # Now find the models
  cur_model = conn.cursor()
  cur_model.execute("SELECT * from model")
  models = cur_model.fetchall()

  for model in models:
    mname = model[0].replace(" ","") 
    # Ignore if the end points are way out!
    new_loop = Loop(mname)
    
    cur_res = conn.cursor()
    cur_res.execute("SELECT * from residue where model='" + mname + "' order by resorder")
    residues = cur_res.fetchall()
  
    # Must have residues to continue
    if len(residues) == 0:
      continue

    temp_residues = []
    for row in residues:
      residue = acids.label_to_amino(row[1])
      reslabel = row[2]
      resorder = row[3]
      temp_residues.append((residue,reslabel,resorder))

    cur_angle = conn.cursor()
    cur_angle.execute("SELECT * from angle where model='" + mname + "' order by resorder")
    angles = cur_angle.fetchall()  
  
    if len(angles) == 0:
      print("ERROR with model " + mname + ". No angles returned")
      continue

    idx = 0
    bad_omega = False
    for row in angles:
      phi = math.radians(row[1])
      psi = math.radians(row[2])
      omega = math.radians(row[3])
      
   
      new_residue = Residue(
          temp_residues[idx][0],
          temp_residues[idx][1], 
          temp_residues[idx][2],
          phi,psi,omega)

      new_loop.add_residue(new_residue)
      idx+=1
    
    final_loops.append(new_loop)
    
  conn.close()
  # return blockers so we can still keep a large set
  
  x = []
  y = []

  for loop in final_loops:
    for res in loop.get_residues():
      if res._phi != 0  and res._psi != 0: 
        x.append(res.phid())
        y.append(res.psid())

  _plot(x,y,"All real AbDb")

