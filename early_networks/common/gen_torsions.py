"""
gen_torsions.py - Generate some data from the PostGreS to torsions
author : Benjamin Blundell
email : me@benjamin.computer
"""

import sys
import time
import psycopg2
import traceback
import math
import random

def _grab_angles_model(conn, mname):
  cur1 = conn.cursor()
  cur1.execute("SELECT * from angle where model='" + mname + "' order by resorder")
  rows = cur1.fetchall()

  aa = []

  if len(rows) == 0:
    return aa

  idx = 0
  for row in rows:
    phi = math.radians(row[1])
    psi = math.radians(row[2])
    omega = math.radians(row[3])

    tt = [phi, psi, omega]

    # -3.0 is our global ignore value
    if idx == 0:
      tt[0] = -3.0
    
    if idx == len(rows)-1:
      tt[1] = -3.0
      tt[2] = -3.0

    aa.append(tt)
    idx+=1
    
  return aa


def grab_angles(conn):
  ''' Grab the angles from the DB, convert to continuous and return '''  
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  angles = {}
  
  for model in models:
    mname = model[0].replace(" ","")
    aa = _grab_angles_model(conn, mname)
    if len(aa) > 0: 
      angles[mname] = {}
      angles[mname]["angles"] = aa
   
  return angles

def _grab_residues_model(conn, mname):
  cur1 = conn.cursor()
  cur1.execute("SELECT * from residue where model='" + mname + "' order by resorder")
  rows = cur1.fetchall()

  aa = []
  for row in rows:
    residue = row[1]
    reslabel = row[2]
    # Changing to sin makes the angle a little more continuous I think
    aa.append( (residue, reslabel) )

  return aa

def grab_residues(conn, angles):
  ''' Grab the residues from the database as well. '''
  cur0 = conn.cursor()
  cur0.execute("SELECT * from model")
  models = cur0.fetchall()
  
  for model in models:
    mname = model[0].replace(" ","")
    aa =_grab_residues_model(conn,mname)

    if mname in angles:
      angles[mname]["residues"] = aa

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
    sys.stdout.write(str(angle[0]) + "," + str(angle[1]) + "," + str(angle[2]) + ", ")

  print()


def gen():
  # Use the prepared abdab db with the correct info
  conn = psycopg2.connect("dbname=pdb_martin user=postgres")
  angles = grab_angles(conn)
  grab_residues(conn, angles)
  return angles


def get_model(mname):
  conn = psycopg2.connect("dbname=pdb_martin user=postgres")
  aa = _grab_angles_model(conn, mname)
  rr = _grab_residues_model(conn, mname)

  angles= {}
  angles["angles"] = aa
  angles["residues"] = rr

  return angles

if __name__ == "__main__":
  angles = gen()
  print_random(angles)
