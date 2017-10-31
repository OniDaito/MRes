"""
gen_data.py - Generate some data from the PostGreS to NN
author : Benjamin Blundell
email : oni@section9.co.uk
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
  
  for model in models:
    mname = model[0].replace(" ","")
    cur1 = conn.cursor()
    cur1.execute("SELECT * from angle where model='" + mname + "' order by resorder")
    rows = cur1.fetchall()

    aa = []

    if len(rows) == 0:
      print("ERROR with model " + mname + ". No angles returned")
      continue

    idx = 0
    for row in rows:
      phi = row[1]
      psi = row[2]
      # Changing to sin makes the angle a little more continuous I think
     
      tt = [ math.sin(math.radians(phi)),
        math.cos(math.radians(phi)), 
        math.sin(math.radians(psi)), 
        math.cos(math.radians(psi))]

      # -3.0 is our global ignore value
      if idx == 0:
        tt[0] = -3.0
      
      if idx == len(rows)-1:
        tt[3] = -3.0

      aa.append(tt)
      idx+=1
      
    angles[mname] = {}
    angles[mname]["angles"] = aa
   
  return angles

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

if __name__ == "__main__":
  angles = gen()
  gtprint_random(angles) 
