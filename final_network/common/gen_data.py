"""
gen_data.py - Generate a set of loop data, ready for batching
author : Benjamin Blundell
email : me@benjamin.computer

This file will read from a postgres or mongo database, assuming
these databases conform to the ones created with build_database.py

"""
import sys, time, traceback, math, random, os
import numpy as np

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.repping import *

# Derived from the endpoint data stored in the database
# TODO - autogen this somehow in our final pipeline
STDDEV = 1.13033939846
LIMIT = 7.537065927577005

class Summary(object):
  ''' Short struct like object that summarises our db access. '''
  def __init__(self):
    self.total = 0
    self.rejected = 0
    self.min_length = 3000
    self.max_length = 0
    self.counts = {}

  def __str__(self):
    ts = "Total: " + str(self.total) + " Rejected: " + str(self.rejected) + " Min: " + str(self.min_length) + " Max: " + str(self.max_length) + "\n"

    for k, v in self.counts.items():
      ts += str(k) + ":" + str(v) + "\n"

    fl = []
    for ky in self.counts.keys():
      for i in range(0,self.counts[ky]):
        fl.append(ky)

    sd = np.std(fl)
    ts += "StdDev: " + str(sd) + " "
    mn = np.mean(fl)
    ts += "Mean: " + str(mn)

    return ts

class Grabber(object):
  ''' A class to pull down data from the various databases and create a selection
  we can pass to the batcher. '''
  def __init__(self, remove_bad_endpoint = False, remove_odd_omega = False, dbname = "pdb", res_limit = 2.2):
    self._db = dbname
    self._user = "postgres"
    self._remove_odd = remove_odd_omega
    self._remove_bad = remove_bad_endpoint
    self._res_limit = res_limit

  def grab(self, limit=200000, min_length=0, max_length=32, block=False, specific="", coords=False):
    ''' Grab performs the actual creation. I've added a way to limit the total
    amount as a way to use the larger db.'''
    return self._grab_postg(limit, min_length,  max_length, block, specific, coords)

  def _grab_postg(self, limit, min_length, max_length, block=False, specific="", coords=False):
    ''' Actually perform the grab with PostGreSQL.'''
    import psycopg2
    summary = Summary()
    conn = psycopg2.connect("dbname=" + self._db + " user=" + self._user) 
    final_loops = []
    rr = 0
    
    # Create a list of redundant items - for now we do not duplicate
    cur_red = conn.cursor()
    cur_red.execute("SELECT * from redundancy")
    reds = cur_red.fetchall()
    reduns = []
    treduns = []

    # TODO - should sort this
    # We need to add the entirety as we can't have any
    # redundant info in test or validate
    # If block is activated we dont store both but remove
    # from all sets anything that appears

    for red in reds:
      if red[1] not in reduns:
        reduns.append(red[1])
      if red[0] not in reduns :
        reduns.append(red[0])
    
      if block:
        if red[1] not in treduns:
          treduns.append(red[1])

    # Now find the models
    cur_model = conn.cursor()
    if len(specific) > 0:
      cur_model.execute("SELECT * from model where code='" + specific + "'")
    else:
      cur_model.execute("SELECT * from model")

    models = cur_model.fetchall()

    for model in models:
      mname = model[0].replace(" ","") 
      # If we want to restrict via resolution here is where we do it
      if self._remove_bad:
        if model[4] > self._res_limit:
          summary.rejected += 1
          continue

      # A redundant item already exists for this so can't add it
      if mname in treduns and block: 
        rr +=1
        continue

      new_loop = Loop(mname)
  
      # Pull out the NeRFed end points 
      #cur_res = conn.cursor()
      #cur_res.execute("SELECT * from nerf where model='" + mname + "'")
      #endpoints = cur_res.fetchall()
      #if len(endpoints) != 1:
      #  summary.rejected += 1
      #  continue
      #endpoint = endpoints[0]
      # Should only be one
      #new_loop._endpoint = [endpoint[1], endpoint[2], endpoint[3]]
       
      cur_res = conn.cursor()
      cur_res.execute("SELECT * from residue where model='" + mname + "' order by resorder")
      residues = cur_res.fetchall()
    
      # Must have residues to continue
      if len(residues) == 0 or len(residues) > max_length or len(residues) < min_length:
        summary.rejected += 1
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
        summary.rejected += 1
        continue

      idx = 0
      bad_omega = False
      for row in angles:
        phi = math.radians(row[1])
        psi = math.radians(row[2])
        omega = math.radians(row[3])
        
        # TODO - 160 is arbitrary and perhaps we should do a proper analysis
        if row[3] < 160.0 and row[3] > -160.0 and row[3] != 0:
          bad_omega = True
    
        new_residue = Residue(
            temp_residues[idx][0],
            temp_residues[idx][1], 
            temp_residues[idx][2],
            phi,psi,omega)

        new_loop.add_residue(new_residue)
        idx+=1
      
      if bad_omega and self._remove_odd:
        summary.rejected += 1
        continue

      # Pull coords from the db
      # With both AbDb and LoopDB we should have the loop required and nothing else
      if coords:
        cur_atom = conn.cursor()
        cur_atom.execute("SELECT * from atom where model='" + mname + "' and (name = 'N' or name = 'CA' or name = 'C') order by serial")
        atoms = cur_atom.fetchall()  

        for row in atoms:
          x = float(row[8]) 
          y = float(row[9])
          z = float(row[10])
          name = row[2] 
          new_atom = Atom(name, x, y, z)
          new_loop.add_atom(new_atom)
          print("NA",new_atom)

      final_loops.append(new_loop)
      
      # Summary
      ll = len(new_loop._residues)
      if ll < summary.min_length:
        summary.min_length = ll
      elif ll > summary.max_length:
        summary.max_length = ll
      
      if not ll in summary.counts.keys():
        summary.counts[ll] = 0
      summary.counts[ll] += 1

      if limit != -1:
        if len(final_loops) >= limit:
          break

    summary.total = len(final_loops)
    conn.close()
    # return blockers so we can still keep a large set
    return (final_loops, summary, reduns)
 
  
if __name__ == "__main__":
  from repping import *
  import acids 
  import argparse
  import pdb
  parser = argparse.ArgumentParser(description='Output a loop and pdb')
  parser.add_argument('--loop', action='store_true', help='Remove bad loops')
  parser.add_argument('modelname', metavar='MODELNAME', type=str, help='the model we want')
  args = parser.parse_args()

  g = Grabber(dbname="loopdb")
  (loops, summary, blockers) = g.grab(max_length=28, specific = args.modelname, coords=True)
  for loop in loops:
    loop.print()
    pdb.loop_to_pdb(args.modelname + ".pdb", loop)
  print(summary)


