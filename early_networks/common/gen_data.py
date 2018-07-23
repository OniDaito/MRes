"""
gen_data.py - Generate some data from the PostGreS to NN
author : Benjamin Blundell
email : me@benjamin.computer
"""

import sys
import time
import psycopg2
import traceback
import math
import random

def grab_angles(conn):
  ''' Grab the angles from the DB, convert to continuous and return '''  
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  angles = {}
  block = False
  
  cur_red = conn.cursor()
  cur_red.execute("SELECT * from redundancy")
  reds = cur_red.fetchall()
  blockers = []
  for red in reds:
    if red[1] not in blockers:
      blockers.append(red[1])
    if red[0] not in blockers:
      blockers.append(red[0])


  for model in models:
    mname = model[0].replace(" ","")
    cur1 = conn.cursor()
    cur1.execute("SELECT * from angle where model='" + mname + "' order by resorder")
    rows = cur1.fetchall() 
    aa = []

    if len(rows) == 0:
      print("ERROR with model " + mname + ". No angles returned")
      continue

    if mname in blockers and block:
      print("Blocked", mname)
      continue

    idx = 0
    for row in rows:
      phi = row[1]
      psi = row[2]
      # Changing to sin makes the angle a little more continuous I think
      #tt = [phi,psi] 
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
      
    angles[mname] = {}
    angles[mname]["angles"] = aa
   
  return (angles, blockers)

def grab_residues(conn, angles):
  ''' Grab the residues from the database as well. '''
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  
  for model in models:
    mname = model[0].replace(" ","")
    cur1 = conn.cursor()
    cur1.execute("SELECT * from residue where model='" + mname + "' order by resorder")
    rows = cur1.fetchall()

    aa = []
    for row in rows:
      residue = row[1]
      reslabel = row[2]
      # Changing to sin makes the angle a little more continuous I think
      aa.append( (residue, reslabel) )
  
    if mname in angles:
      angles[mname]["residues"] = aa

def create_data_sets(angles, blockers = []):
  ''' Create our various datasets. 80% train, 10% test, 10% validation. ''' 
  a = len(angles)
  k = list(angles.keys())
  training_labels = []
  test_labels = []
  validation_labels = []
  non_redun_set = []

  redun_set = []
  for name in k:
    if name not in blockers:
      non_redun_set.append(name)
    else:
      redun_set.append(name)

  print ("Redundant / Non redundant", len(redun_set), len(non_redun_set))

  tl = min( int(math.floor(len(non_redun_set)/2)), int(float(a) / 100.0 * 10.0))
  for n in range(0,tl):
    r = random.randint(0,len(non_redun_set)-1)
    validation_labels.append(non_redun_set.pop(r))

  tl = min(int(math.floor(len(non_redun_set) / 2)), int(float(a) / 100.0 * 10.0))
  for n in range(0,tl):
    r = random.randint(0,len(non_redun_set)-1)
    test_labels.append(non_redun_set.pop(r))

  training_labels = redun_set

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
  (angles, blockers) = grab_angles(conn)
  grab_residues(conn, angles)
  return (angles, blockers)

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
  (angles, blockers) = gen()
  gtprint_random(angles) 
