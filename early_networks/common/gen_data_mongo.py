"""
gen_data_mongo.py - Generate some data from the mongodb to NN
author : Benjamin Blundell
email : me@benjamin.computer
"""

#TODO - this might end up being too much as the data is really large

import sys
import time
import traceback
import math
import random
import pymongo

from pymongo import MongoClient

def grab_angles(conn):
  ''' Grab the angles from the DB, convert to continuous and return.
  Going with a 31 max for now.'''  
  db = conn.smallmodels
  cursor = db.find(no_cursor_timeout=True)
  angles = {}
  
  for model in cursor:
    mname = model["code"].replace(" ","")
    db2 = conn.angles
    cur1 = db2.find({"model" : mname}, no_cursor_timeout=True).sort("resorder", pymongo.ASCENDING) 
    aa = []

    if cur1.count() == 0:
      print("ERROR with model " + mname + ". No angles returned")
      continue

    idx = 0
    for row in cur1:
      phi = row["phi"]
      psi = row["psi"]
      # Changing to sin makes the angle a little more continuous I think
     
      tt = [ math.sin(math.radians(phi)),
        math.cos(math.radians(phi)), 
        math.sin(math.radians(psi)), 
        math.cos(math.radians(psi))]

      # -3.0 is our global ignore value
      if idx == 0:
        tt[0] = -3.0
        tt[1] = -3.0
  
      aa.append(tt)
      idx+=1
   
    # Check any of the results for NaN which happens apparently
    bad = False
    for tt in aa:
      for ta in tt:
        if math.isnan(ta):
          bad = True
          break

    if not bad:
      aa[len(aa)-1][2] = -3.0
      aa[len(aa)-1][3] = -3.0
      angles[mname] = {}
      angles[mname]["angles"] = aa

  return angles

def grab_residues(conn, angles):
  ''' Grab the residues from the database as well. '''
  db = conn.smallmodels
  cur0 = db.find(no_cursor_timeout=True)
  acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO","SEC", "SER", "THR", "TYR", "TRP", "VAL"]
  
  for model in cur0:
    mname = model["code"].replace(" ","")
    db2 = conn.residues
    cur1 = db2.find({"model" : mname}, no_cursor_timeout=True).sort("resorder", pymongo.ASCENDING)
    aa = []

    bad = False
    for row in cur1:
      residue = row["residue"]
      reslabel = row["reslabel"]
      # We need to check and see if any of the residues are 'GLX', 'CSO', 'ASX', or 'UNK' as this
      # happens apparently
      if residue not in acids: 
        bad = True
        break
      
      # Changing to sin makes the angle a little more continuous I think
      aa.append( (residue, reslabel) )
    
    if not bad:
      if mname in angles:
        angles[mname]["residues"] = aa
    else:
      angles.pop(mname, None)

def create_data_sets(angles):
  ''' Create our various datasets. 80% train, 10% test, 10% validation. ''' 
  a = len(angles)
  k = list(angles.keys())
  training_labels = []
  test_labels = []
  validation_labels = []
  tl = int(float(a) / 100.0 * 80.0)

  for n in range(0,tl):
    r = random.randint(0,len(k)-1)
    training_labels.append(k.pop(r))

  tl = int(float(a) / 100.0 * 10.0)
  
  for n in range(0,tl):
    r = random.randint(0,len(k)-1)
    test_labels.append(k.pop(r))

  validation_labels = k

  training = {}
  test = {}
  validation = {}

  for t in training_labels:
    training[t] = angles[t]

  for t in test_labels:
    test[t] = angles[t]

  for t in validation_labels:
    validation[t] = angles[t]

  return (training, validation, test)

def print_random(angles):
  ''' Print out a random CDR-H3 '''
  k = list(angles.keys())
  r = random.randint(0,len(k)-1)
  kk = k[r]

  print("*** " + kk + " ***")
  for residue in angles[kk]['residues']:
    sys.stdout.write(residue[0] + ",")

  print()

  for angle in angles[kk]['angles']:
    sys.stdout.write(str(angle[0]) + "-" + str(angle[1]) + ":" + str(angle[2]) + "-" +str(angle[3]) + ", ")

  print()

def gen():
  conn = MongoClient()
  conn = conn.pdb_loopdb
  angles = grab_angles(conn)
  grab_residues(conn, angles)
  return angles

if __name__ == "__main__":
  angles = gen()
  print_random(angles) 
