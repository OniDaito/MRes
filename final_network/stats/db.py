"""
db.py - Grab the loops from the DB
author : Benjamin Blundell
email : me@benjamin.computer

"""

import os, sys, signal, subprocess, math
from Bio.PDB import *
import numpy as np

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.repping import *
  from common.gen_data import Summary

class Grabber(object):
  ''' A class to pull down data from the various databases and create a selection
  we can pass to the batcher. '''
  def __init__(self, mongo=False):
    self._use_mongo = mongo
    self._db = "pdb_martin"
    self._user = "postgres"

  def grab(self, mname):
    if self._use_mongo:
      return self._grab_mongo(mname)
    else:
      return self._grab_pg(mname)

  def _grab_mongo(self, mname):
    ''' Perform the actual grab from a mongo database.'''
    import pymongo
    from pymongo import MongoClient
    
    conn = MongoClient()
    conn = conn.pdb_loopdb # TODO should set name here I think
   
    db_angles = conn.angles
    cur_angles = db_angles.find({"model" : mname}, no_cursor_timeout=True).sort("resorder", pymongo.ASCENDING) 
    aa = []

    if cur_angles.count() == 0:
      print("ERROR with model " + mname + ". No angles returned")
      return

    bad_data = False
    idx = 0
    temp_angles = []
    for row in cur_angles:
      phi = math.radians(row["phi"])
      psi = math.radians(row["psi"])
      omega = math.radians(row["omega"])

      # NaN can apparently sneak in here and cause us issues
      if math.isnan(phi) or math.isnan(psi) or math.isnan(omega):
        print("ERROR with model " + mname + ". NaN returned")
        return

      temp_angles.append((phi,psi,omega))
 
    new_loop = Loop(mname)      
    idx = 0
    db_residues = conn.residues
    cur_res = db_residues.find({"model" : mname}, no_cursor_timeout=True).sort("resorder", pymongo.ASCENDING)

    for row in cur_res:
      reslabel = row["reslabel"]
      resorder = row["resorder"]
      # We need to check and see if any of the residues are 'GLX', 'CSO', 'ASX', or 'UNK' as this  
      resname = acids.label_to_amino(row["residue"])
      if resname == None:
        print("ERROR with model " + mname + ". Error in residue")
        return

      new_residue = Residue(
          resname, resorder, reslabel,
          temp_angles[idx][0], 
          temp_angles[idx][1], 
          temp_angles[idx][2])

      new_loop.add_residue(new_residue)
      idx +=1

    db_atoms = conn.atoms
    cur_atoms = db_atoms.find({ "$and" :[ {"model": mname }, { "$or" : [{"name": "N"}, {"name" : "CA"}, {"name" : "C"} ]}]}, no_cursor_timeout=True).sort("serial",pymongo.ASCENDING)

    # I believe we need to remove the first and last three residues right? Atoms is a 
    # complete list including the roots

    for row in cur_atoms:
      x = float(row["x"]) 
      y = float(row["y"])
      z = float(row["z"])
      name = row["name"] 
      new_atom = Atom(name, x, y, z)
      new_loop.add_atom(new_atom)

    new_loop._backbone = new_loop._backbone[9:-9]

    if len(new_loop._backbone) == 0:
      return None

    return new_loop

  def _grab_pg(self, mname):
    ''' Actually perform the grab with PostGreSQL.'''
    import psycopg2
    summary = Summary()
    conn = psycopg2.connect("dbname=" + self._db + " user=" + self._user) 
    final_loops = []
    cur_model = conn.cursor()
    cur_model.execute("SELECT * from model where code='" + mname + "'")
    models = cur_model.fetchall()
    new_loop = Loop(mname)

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
  
    idx = 0
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

    cur_atom = conn.cursor()
    cur_atom.execute("SELECT * from atom where model='" + mname + "' and (name = 'N' or name = 'CA' or name = 'C') and chainid='H' and resseq >= 95 and resseq<= 102 order by serial")
    atoms = cur_atom.fetchall()  

    for row in atoms:
      x = float(row[8]) 
      y = float(row[9])
      z = float(row[10])
      name = row[2] 
      new_atom = Atom(name, x, y, z)
      new_loop.add_atom(new_atom)
      
    conn.close()
    return new_loop
