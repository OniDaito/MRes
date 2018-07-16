"""
gen_data_vec.py - Generate some data from the PostGreS to Vectors
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
  vectors = {}
  
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
  
      y = math.sin(math.radians(psi))
      tt = math.cos(math.radians(psi))
      x = tt * math.cos(math.radians(phi)) 
      z = tt * math.sin(math.radians(phi))
      
      tt = [x,y,z]
       
      aa.append(tt)
      idx+=1
      
    vectors[mname] = {}
    vectors[mname]["vectors"] = aa
   
  return vectors

def grab_residues(conn, vectors):
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
  
    if mname in vectors:
      vectors[mname]["residues"] = aa

def create_data_sets(vectors):
  ''' Create our various datasets. 80% train, 10% test, 10% validation. ''' 
  a = len(vectors)
  k = list(vectors.keys())
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
    training[t] = vectors[t]

  for t in test_labels:
    test[t] = vectors[t]

  for t in validation_labels:
    validation[t] = vectors[t]

  return (training, validation, test)

def print_random(vectors):
  ''' Print out a random CDR-H3 '''
  k = list(vectors.keys())
  r = random.randint(0,len(k)-1)
  kk = k[r]

  print("*** " + kk + " ***")
  for residue in vectors[kk]['residues']:
    sys.stdout.write(residue[0] + ",")

  print()

  for vector in vectors[kk]['vectors']:
    sys.stdout.write(str(vector[0]) + "-" + str(vector[1]) + "-" + str(vector[2]) + ", ")

  print()

def gen():
  # Use the prepared abdab db with the correct info
  conn = psycopg2.connect("dbname=pdb_martin user=postgres")
  vectors = grab_angles(conn)
  grab_residues(conn, vectors)
  return vectors

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
