"""
pdb_to_postgres.py - convert a series of PDB loops to data for postgres
author : Benjamin Blundell
email : me@benjamin.computer

Currently using biopython for this but I don't think we *really* need
to. Some PDBs actually don't contain torsion angles so we need to
detect these I think and reject.

By default, we cut out the 3 residues prior to the loop and the 3 
after. We do the same with both LoopDB and AbDb.

"""

from Bio.PDB import *
from Bio.PDB.Polypeptide import is_aa, three_to_one
import sys
import time
import psycopg2
import traceback
import os
import argparse
import torsions

class PDBDB ():
  def __init__(self):
    self.parser = PDBParser()
    self.conn = None

  def db_connect(self, dbname):
    """ Connect to the postgresql database instance. """
    self.conn = psycopg2.connect("dbname=" + dbname + " user=postgres")
  
  def gen_angles(self):
    """ Generate the angles we will need for our neural net. """ 
    error_models = []
    if self.conn != None:
      cur = self.conn.cursor()
 
    try:
      cur.execute("SELECT * FROM model")

      for model in cur.fetchall():
        mname = model[0].replace(" ","")
        print("Generating angles for", mname)
        cur.execute("SELECT * FROM atom where model = '" + mname + "' order by serial")
        atoms = cur.fetchall()
   
        try: 
          # Extract the different residues 
          residues = []
          current_residue = [] # name, amino, num, x, y, z
          eurrent_name = ""
          dd = 0
          previdx = atoms[6]

          for atom in atoms:
            # model, serial name altloc resname chainid resseq icode x y z occupancy tempf element id 
          
            if atom[4] != current_name or atom[2] == 'N':
              current_name = atom[4]
              if len(current_residue) != 0 :
                if int(atom[6]) == previdx:
                  dd += 1

                previdx = int(atom[6])
                residues.append(current_residue.copy())
                current_residue = []
                dd = 0
        
            rr = (atom[2], atom[4], int(atom[6]),\
                float(atom[8]), float(atom[9]), float(atom[10]))
            current_residue.append(rr)

          residues.append(current_residue)
          # Previously we dropped 3 residues either side. We now include
          # the entire PDB file and drop afterwards if we so choose.
          if len(residues) < 3:
            continue

          # Write the residues we are considering into the db for speed
          idx = 0
          for res in residues:
            try:
              cur.execute("INSERT INTO residue (model,residue,\
                  reslabel, resorder) VALUES (%s, %s, %s, %s)",\
                  (mname,res[1][1],res[1][2], idx))
              self.conn.commit()
              
            except Exception as e:
              print("Error inserting into DB", model, e)
              traceback.print_exc()
              error_models.append(model)

            idx+=1
          
          # Now we have all the residues, work out the angles and write
          # to the database.
          angles = torsions.derive_angles(residues)
          idx = 0
          for aa in angles:
            try:
             cur.execute("INSERT INTO angle (model,phi,psi,omega,\
                 resorder) VALUES (%s, %s, %s, %s, %s)",\
                 (mname, aa[0], aa[1], aa[2], idx))
             self.conn.commit()
  
            except Exception as e:
              print("Error inserting into DB - " + model, e)
              error_models.append(model)
            idx +=1
          
          sys.stdout.write("\n")
        except Exception as e:
          print("Error with model: ", model)
          print(e)
          error_models.append(model)
          traceback.print_exc()
          #sys.exit()

    except Exception as e:
        print ("Exception in DB read: ", e)
        traceback.print_exc()
        sys.exit()
    
    with open("error_models.txt", 'w') as w:
      for item in error_models:
        w.write(str(item) + "\n")
  
  def _check(self, model):
    seq = ""
    for chain in model:
      chainID = chain.get_id()
      # Assuming there is only one chain! Bit naughty!
      for residue in chain:
        ## The test below checks if the amino acid
        ## is one of the 20 standard amino acids
        ## Some proteins have "UNK" or "XXX", or other symbols
        ## for missing or unknown residues
        if is_aa(residue.get_resname(), standard=True):
          seq += three_to_one(residue.get_resname())
        else:
          return (False, seq)

    atom_count = 0
    for atom in model.get_atoms():
      atom_count +=1
      if atom_count > 10:
        break
        
    if atom_count < 10:
      print("PDB Model contains no atoms.")
      return (False, seq)

    return (True, seq)
    
  def process_pdb(self, pdb_path, complete_model=False):
    ''' Read a PDB and write to the DB.'''
    try:
      bn = os.path.basename(pdb_path),
      st = self.parser.get_structure(bn, pdb_path)
    except:
      print("Failed to parse:", pdb_path)
      return

    models = st.get_models()
    for model in models:
      mname = os.path.splitext(model.get_full_id()[0][0])[0]
      print("Working with ", mname)
      tokens = mname.split("_")
      model_code = tokens[0]
      loop_code = mname 

      try:
        # Some PDBs are missing atoms so make sure we have some :/
        # This code is a bit rough but might work
        # We perform a lot of checks here to see if the PDB is valid
        (res, seq) = self._check(model)   
        if not res:
          continue

        # We make a call to get the mmCIF file which we need to make
        # this work
        pdbl = PDBList()      
        dn = "" + filename[1].lower() + filename[2].lower() + "/" + model_code.lower() + ".cif"
  
        if not os.path.isfile(dn):
          pdbl.retrieve_pdb_file(model_code.upper())

        resolution = -1.0
        rvalue = -1.0
        rfree = -1.0
        try: 
          with open(dn, 'r') as f:
            pdbraw = f.readlines()  
            try:
              for line in pdbraw:
                if "_refine.ls_R_factor_R_work " in line:
                  tokens = line.split(" ")
                  rvalue = float(tokens[-2])

                elif "_refine.ls_R_factor_R_free " in line:
                  tokens = line.split(" ")
                  rfree=float(tokens[-2])
         
                elif "_reflns.d_resolution_high " in line:
                  tokens = line.split(" ")
                  resolution=float(tokens[-2])
            except Exception as e:
              print("Error parsing R values", e)
        except Exception as e:
          print("Failed to read R file", e)

        if self.conn != None:
          cur = self.conn.cursor()
          cur.execute("SELECT FROM model where code = '"\
              + loop_code + "'")
          tt = cur.fetchone()

          if tt == None:
            if self.conn != None:
              cur.execute("INSERT INTO model (code, filename, \
                  rvalue, rfree, resolution) \
                  VALUES (%s, %s, %s, %s, %s)", \
                  (loop_code, bn, rvalue, rfree, resolution))
              self.conn.commit()
 
            for atom in model.get_atoms():
              # If we have the complete model, I.e AbDb and not loopdb
              # we need to extract atoms belonging to the range 3 either
              # side of 95 to 102 on the heavy chain.

              #import pdb; pdb.set_trace()
              #print(atom, dir(atom))
              res = atom.get_parent()
              chain = res.get_parent()
              aserial = int(atom.get_serial_number())
              aname = atom.get_name()
              aaltloc = atom.get_altloc()
              aresname = res.resname
              achainid = chain.id
              aresseq = int(res.id[1])
              ax = float(atom.get_coord()[0])
              ay = float(atom.get_coord()[1])
              az = float(atom.get_coord()[2])
              aocc = float(atom.get_occupancy())
              abfac = float(atom.get_bfactor())
              aele = atom.element
        
              if self.conn != None:
                # Test to see if we have chains and what not, keeping 3 either side
                if not complete_model or (achainid == "H" and aresseq >= 92 and aresseq <= 105):
                  cur.execute("INSERT INTO atom (model, serial, name, \
                    altloc, resname, chainid, resseq, x, y, z, \
                    occupancy, tempfactor, element) \
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                    %s, %s, %s)", \
                    (loop_code, aserial, aname, aaltloc, aresname, \
                    achainid, aresseq, ax, ay, az, aocc, abfac, \
                    aele)) 
                  self.conn.commit()

      except Exception as e:
        print ("Exception in model insert: ", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

  def redundancy(self, rfile = None):
    """ Generate the redundancy data. If the sequence is the same
    or the loops overlap, don't use it."""
    from itertools import groupby
    if self.conn == None:
      return

    duplicates = []
    
    # Firstly, if rfile is not None, we can read it and add duplicates from that
    if rfile != None:
      with open(rfile, 'r') as f:
        cur = self.conn.cursor()
        for line in f.readlines():
          tokens = line.replace(" ","").split(",")
          for token in tokens[1:]:
            token = token.replace(" ","")
            # Simple test to make sure we have a good name
            if len(token) > 4:
              # Make sure the token represents a model that exists
              cur.execute("SELECT * FROM model where code = '" + token + "'")
              mods = cur.fetchall()
              if len(mods) != 0:
                duplicates.append((tokens[0], token))

    cur = self.conn.cursor()
    cur.execute("select * from model")
    models = cur.fetchall()
    for idx in range(0, len(models)):
      model = models[idx]

      for jdx in range(idx+1, len(models)):
        check = models[jdx]
 
        # First, check if there are residue markers in the filename
        if check[0] != model[0]:
          if "_" in model[0]:
            tokens_m = model[0].replace(" ","").split("_")
            tokens_c = check[0].replace(" ","").split("_")
            m_name = tokens_m[0]
            c_name = tokens_c[0]

            if m_name == c_name:
              m_code = [''.join(g) for _, g in groupby(tokens_m[1], str.isalpha)]          
              c_code = [''.join(g) for _, g in groupby(tokens_c[1], str.isalpha)]          
              
              if m_code[0] == c_code[0] and m_code[2] == c_code[2]:
                ms = int(m_code[1])
                me = int(m_code[3])
                cs = int(c_code[1])
                ce = int(c_code[3])
                if (ms >= cs and me <= ce) or (cs >= ms and ce <= me):
                  if not ((check[0], model[0]) in duplicates or (model[0], check[0]) in duplicates) :
                    # Its a duplicate so add to list of dups
                    print(tokens_m, tokens_c)
                    duplicates.append((model[0], check[0]))
    ii = 0
    for (c,m) in duplicates:
      cur.execute("SELECT * FROM redundancy where model = '" + m + "' and match = '" + c + "'")
      dups = cur.fetchall()
      if len(dups) == 0:
        cur.execute("INSERT INTO redundancy (model, match) VALUES (%s, %s)", (c, m))
        ii += 1
        if ii >= 10:
          self.conn.commit()
          ii = 0

    # We should also check the residue lists to make sure they aren't identical 
    cur.execute("select * from model")
    models = cur.fetchall()
    
    for idx in range(0, len(models)):
      model = models[idx]

      for jdx in range(idx+1, len(models)):
        check = models[jdx]
        if check[0] != model[0]:
          if "_" in model[0]:
            tokens_m = model[0].replace(" ","").split("_")
            tokens_c = check[0].replace(" ","").split("_")
            m_name = model[0].replace(" ","")
            c_name = check[0].replace(" ","")
          
            if not ((c_name, m_name) in duplicates or (m_name, c_name) in duplicates) :
              cur2 = self.conn.cursor()
              cur2.execute("SELECT * FROM residue where model = '" + m_name + "' order by resorder")
              m_res = []
              residues = cur2.fetchall()
              for res in residues:
                m_res.append(res[1])

              cur2.execute("SELECT * FROM residue where model = '" + c_name + "' order by resorder")
              c_res = []
              residues = cur2.fetchall()
              for res in residues:
                c_res.append(res[1])

              if m_res == c_res:
                # duplicate residues so add to redundancy
                print(tokens_m, tokens_c)
                duplicates.append((m_name, c_name))
     
    ii = 0 
    for (c,m) in duplicates:
      cur.execute("SELECT * FROM redundancy where model = '" + m + "' and match = '" + c + "'")
      dups = cur.fetchall()
      if len(dups) == 0:
        cur.execute("INSERT INTO redundancy (model, match) VALUES (%s, %s)", (c, m))
        ii += 1
        if ii >= 10:
          self.conn.commit()
          ii = 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process arguments")
  parser.add_argument('db_name', metavar='db_name',\
      help='The destination database name.')
  parser.add_argument('dir_name', metavar='dir_name',\
      help='The directory with the PDBs we want to process.')
  parser.add_argument('--rfile', dest='rfile',
      help='The list of redundant PDB files - used with abdb.')
  parser.add_argument('--dry-run', dest='dry_run',\
      action='store_true', default=False,\
      help='Do not enter new data into the db. Default False.')
  parser.add_argument('--full-model', dest='complete_model',\
      action='store_true', default=False,\
      help='PDB files contain more than just the loop. Default False.')
 
  parser.add_argument('--limit', dest='limit',\
      type=int, default=-1,\
      help='Limit the number of pdb files. Default unlimited.')

  args = vars(parser.parse_args())
  p = PDBDB()
  print(args)
  if not args['dry_run']:
    p.db_connect(args["db_name"])
  
  count = 0
  for dirname, dirnames, filenames in os.walk(args["dir_name"]):
    for filename in filenames:
      p.process_pdb(os.path.join(dirname, filename), complete_model=args['complete_model'])
      count = count + 1
      if args['limit'] != -1 and count >= args['limit']:
        p.gen_angles()
        p.redundancy(args["rfile"])
        sys.exit(0)
  
  p.gen_angles()
  p.redundancy(args["rfile"])
     
