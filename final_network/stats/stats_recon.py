"""
stats_recon.py
author : Benjamin Blundell
email : me@benjamin.computer

Test how good NeRF and Martin's reconstruction programs are

"""
import os, sys, signal, subprocess, math
from Bio.PDB import *
import numpy as np

# Import our shared util
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
import common.acids as acids

def do_stats(names, loops, reals, do_omega = True):
  ''' Perform some stats on our three loop types.'''
  import common.nerf as nerf
  import common.pdb as pdb
  
  idx = 0
  pairs = []
  for loop in loops:
    nf = nerf.NeRF()
    coords = nf.compute_positions_loop(loop)
    name = names[idx]
    n0 = name + "_nerf.pdb"
    pdb.coords_to_pdb( n0, name, loop._residues, coords)
    n1 = name + "_real.pdb"
    pdb.coords_to_pdb( n1, name, loop._residues, reals[idx])
    n2 = name + "_mrtn.pdb"
    gen_martin(loop, name, do_omega)
    pairs.append((n0,n1,n2)) 
    idx += 1

  rmsds = []
  rmsdm = []
  for trio in pairs:
    try:
      pro = subprocess.run(["pdbfit", trio[0], trio[1] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      tr = float(tr.replace("RMSD  ",""))
      rmsds.append(tr)
    
      if tr > 1.0:
        print("NeRF", trio[1], tr)

      pro = subprocess.run(["pdbfit", "-w", trio[0], trio[1] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      with open(trio[1],"w") as f:
        f.write(tr)

      pro = subprocess.run(["pdbfit", trio[1], trio[2] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      tr = float(tr.replace("RMSD  ",""))
      rmsdm.append(tr)
      
      if tr > 1.0:
        print("Mrtn", trio[1], tr)

      pro = subprocess.run(["pdbfit", "-w", trio[1], trio[2] ], stdout=subprocess.PIPE)
      tr = pro.stdout.decode()
      with open(trio[2],"w") as f:
        f.write(tr) 

    except:
      print("Failed pdbfit on", trio[1])
 
  print("NeRF average", sum(rmsds) / len(rmsds))  
  print("Martin average", sum(rmsdm) / len(rmsdm))

def pad(ss):
  pad = ""
  for i in range(0,8-len(ss)):
    pad += " "
  return pad + ss

def gen_martin(loop, name, do_omega = True):
  ''' Create the required output files for genloop.'''
  with open("seq.martin","w") as f:
    f.write("(")
    for res in loop._residues:
      f.write(acids.amino_to_letter(res._name))
    f.write(")\n")
  
  with open("torsion.martin","w") as f:
    f.write(str(len(loop._residues)))
    f.write(" 1\n\n")
    f.write("title line\n")
    f.write("----------------------------------\n")

    for res in loop._residues:
      f.write("   " + acids.amino_to_letter(res._name))
      phi = pad("{0:.3f}".format(res.phid()))
      psi = pad("{0:.3f}".format(res.psid()))
      omega = pad("{0:.3f}".format(res.omegad()))
      if not do_omega:
        omega = pad("{0:.3f}".format(180.0))
      f.write("    " + phi + " " + psi + " " + omega + "\n")
  
  try:
    pro = subprocess.run(["genloop", "seq.martin", "torsion.martin", name + "_mrtn.pdb" ], stdout=subprocess.PIPE)
  except:
    print("Failed genloop on", name)

def gen_reals(names):
  ''' Given a set of model names, find the real atom positions.'''
  import psycopg2
  _db = "pdb_martin"
  _user = "postgres"
  conn = psycopg2.connect("dbname=" + _db + " user=" + _user)  
  model_coords = []

  for name in names:
    mname = name.replace(" ","") 
    cur_res = conn.cursor()
    cur_res.execute("SELECT * from atom where chainid='H' and resseq >= 95 and resseq <= 102 and model='" + mname + "' and (name='C' or name='CA' or name='N') order by serial")
    atoms = cur_res.fetchall()
    coords = []
    for atom in atoms:
      coords.append((atom[8], atom[9], atom[10]))
    
    model_coords.append(coords)

  conn.close()
  return model_coords

def gen_loops(limit=-1, do_omega=True):
  ''' Generate the loops from our database.'''
  import psycopg2
  from common.gen_data import Loop, Residue

  _db = "pdb_martin"
  _user = "postgres"
  conn = psycopg2.connect("dbname=" + _db + " user=" + _user) 
  final_loops = []
  names = []
  cur_model = conn.cursor()
  cur_model.execute("SELECT * from model order by code")
  models = cur_model.fetchall()

  for model in models:
    mname = model[0].replace(" ","")   
    new_loop = Loop(mname)
    
    # Pull out the NeRFed end points 
    cur_res = conn.cursor()
    cur_res.execute("SELECT * from nerf where model='" + mname + "'")
    endpoints = cur_res.fetchall()
    if len(endpoints) != 1:
      continue
    endpoint = endpoints[0]
    # Should only be one
    new_loop._endpoint = [endpoint[1], endpoint[2], endpoint[3]]
     
    cur_res = conn.cursor()
    cur_res.execute("SELECT * from residue where model='" + mname + "' order by resorder")
    residues = cur_res.fetchall()
    
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
    for row in angles:
      phi = math.radians(row[1])
      psi = math.radians(row[2])
      omega = math.radians(row[3])
      if not do_omega:
        omega = math.pi

      new_residue = Residue(
          temp_residues[idx][0],
          temp_residues[idx][1], 
          temp_residues[idx][2],
          phi,psi,omega)

      new_loop.add_residue(new_residue)
      idx+=1
    #print("Generated",mname)    
    names.append(mname)
    if limit != -1:
      if len(final_loops) >= limit:
        break

    final_loops.append(new_loop)
  
  conn.close()
  return (final_loops, names)

if __name__ == "__main__":
  (loops, names) = gen_loops(limit=-1, do_omega=False)
  reals = gen_reals(names)
  do_stats(names, loops, reals)
