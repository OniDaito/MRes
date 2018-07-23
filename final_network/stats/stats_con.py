"""
stats_con.py - Generate new pairs depending on how
well they are reconstructed.
author : Benjamin Blundell
email : me@benjamin.computer

"""

import os, sys, signal, subprocess, math

from common.repping import *
  from Bio.PDB import *
  import numpy as np
  from db import Grabber
  from copy import deepcopy
  from common import pdb

  g = Grabber()
  real_loop = g.grab("1U8K_1")
  print(real_loop)
  print("Num Atoms:", len(real_loop._backbone))
  for atom in real_loop._backbone:
    print(atom)

  print ("Computed backbone")
  comp_loop = deepcopy(real_loop)

  from common import nerf
  nf = nerf.NeRF()
  del comp_loop._backbone[:]
  nf.gen_loop_coords(comp_loop, omega=False)
  
  for atom in comp_loop._backbone:
    print(atom)

  pdb.loop_to_pdb("loop_real.pdb", real_loop)
  pdb.loop_to_pdb("loop_comp.pdb", comp_loop)

  try:
    pro = subprocess.run(["pdbfit", "-c", "loop_real.pdb", "loop_comp.pdb" ], stdout=subprocess.PIPE)
    tr = pro.stdout.decode()
    print(tr)
    pro = subprocess.run(["pdbfit", "-c", "-w", "loop_real.pdb", "loop_comp.pdb" ], stdout=subprocess.PIPE)
    tr = pro.stdout.decode()
    with open("loop_algn.pdb","w") as w:
      w.write(tr)
  except Exception as e:
    print(e)
