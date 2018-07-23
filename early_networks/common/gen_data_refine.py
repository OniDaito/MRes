"""
gen_data_end.py - Generate some data from the PostGreS to NN, restricting via endpoint dist
author : Benjamin Blundell
email : me@benjamin.computer
"""

import sys
import time
import psycopg2
import traceback
import math
import random

STDDEV = 1.13033939846
LIMIT = 7.537065927577005

def grab_angles(conn):
  ''' Grab the angles from the DB, convert to continuous and return '''  
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  angles = {}

  for model in models:
    mname = model[0].replace(" ","")
    cur1 = conn.cursor()

    if model[3] == None:
      print("rejected", mname)
      continue

    dist = float(model[3])

    # limiting via the stats we found
    if dist > (STDDEV + LIMIT) or dist < (LIMIT - STDDEV):
      print("rejected", mname)
      continue
   
    cur1.execute("SELECT * from angle where model='" + mname + "' order by resorder")
    rows = cur1.fetchall()    
    aa = []

    if len(rows) == 0:
      print("ERROR with model " + mname + ". No angles returned")
      continue

    idx = 0
    bad_omega = False
    for row in rows:
      phi = row[1]
      psi = row[2]
      # Changing to sin makes the angle a little more continuous I think
      omega = row[3]
      
      # check for omega - quit out if we get an odd value
      # Give it 30 degrees either side.
      if omega < 150.0 and omega > -150.0 and omega != 0:
        bad_omega = True
        continue

      tt = [ math.sin(math.radians(phi)),
        math.cos(math.radians(phi)), 
        math.sin(math.radians(psi)), 
        math.cos(math.radians(psi))]

      # -3.0 is our global ignore value
      if idx == 0:
        tt[0] = -3.0
        tt[1] = -3.0
      
      if idx == len(rows)-1:
        tt[2] = -3.0
        tt[3] = -3.0
        
      aa.append(tt)
      idx+=1
   
    # Grab the endpoints here as well
    cur1.execute("SELECT * from nerf where model='" + mname + "'")
    rows = cur1.fetchall()    
    nn = [rows[0][1], rows[0][2], rows[0][3]]

    if not bad_omega:      
      angles[mname] = {}
      angles[mname]["angles"] = aa
      angles[mname]["endpoint"] = nn
      print("accepted", mname)

  return angles

def grab_residues(conn, angles):
  ''' Grab the residues from the database as well. '''
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  
  for model in models:
    mname = model[0].replace(" ","")
    cur1 = conn.cursor()

    if mname in angles:
      cur1.execute("SELECT * from residue where model='" + mname + "' order by resorder")
      rows = cur1.fetchall()
      aa = []

      for row in rows:
        residue = row[1]
        reslabel = row[2]
        # Changing to sin makes the angle a little more continuous I think
        aa.append( (residue, reslabel) )

      angles[mname]["residues"] = aa

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
  # Use the prepared abdab db with the correct info
  conn = psycopg2.connect("dbname=pdb_martin user=postgres")
  angles = grab_angles(conn)
  grab_residues(conn, angles)
  return angles

def get_model_ca(mname):
  # return the coords for CA backbone
  conn = psycopg2.connect("dbname=pdb_martin user=postgres")
  cur1 = conn.cursor()
  cur1.execute("SELECT * from atom where model='" + mname + "' and chainid = 'H' and resseq >= 95 and resseq <= 102 order by serial")
  rows = cur1.fetchall()
  rt = []
  for row in rows:
    if "CA" in row[2]:
      rt.append(row)
      #print(str(row[8]) + ", " + str(row[9]) + ", " + str(row[10]))
  return rt

if __name__ == "__main__":
  angles = gen()
  print_random(angles) 
