"""
pdb.py - Functions to read and write to PDB files
author : Benjamin Blundell
email : me@benjamin.computer

"""

import numpy as np
import math, os
from Bio.PDB import *

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common import acids
  from common import repping

def loop_to_pdb(filepath, loop):
  ''' Given XYZ tuples of Nitrogen, Carbon alpha and Carboxyl Carbon positions (in that order), and the residues, write to a file. '''
  pdb_str = printpdb(loop)
  with open(filepath, 'w') as f:
    f.write(pdb_str)

def loop_from_pdb(pdb_path):
  ''' Simple PDB that assumes just one model and chain, of a single loop.''' 
  parser = PDBParser()
  
  try:
    bn = os.path.basename(pdb_path),
    st = parser.get_structure(bn, pdb_path)
  except Exception as e:
    print("Failed to parse:", pdb_path, e)
    return

  models = st.get_models()
  loop = repping.Loop("l33t")

  for model in models:
    mname = os.path.splitext(model.get_full_id()[0][0])[0]
    loop._name = mname
    try:
      # Some PDBs are missing atoms so make sure we have some :/
      # This code is a bit rough but might work
      atom_count = 0

      for atom in model.get_atoms():
        atom_count +=1
        if atom_count > 10:
          break
      
      if atom_count < 3:
        print("PDB Model contains no atoms.")
        continue

      for atom in model.get_atoms():
        res = atom.get_parent()
        chain = res.get_parent()

        aserial = int(atom.get_serial_number())
        aname = atom.get_name()
        aresname = res.resname
        achainid = chain.id
        aresseq = int(res.id[1])
          
        if aname == "N" or aname == "C" or aname == "CA":
          ax = float(atom.get_coord()[0])
          ay = float(atom.get_coord()[1])
          az = float(atom.get_coord()[2])
          new_atom = repping.Atom(aname, ax, ay, az )
          loop.add_atom(new_atom)

      idx = 0
      for residue in model.get_residues():
        name = acids.label_to_amino(residue.get_resname()) 
        loop.add_residue(repping.Residue(name, idx, str(idx), 0, 0, 0))
        idx += 1

    except Exception as e:
      print ("Exception creating loop from pdb.", e)

    return loop

def printpdb(loop):  
  ''' Create the PDB file string. Assume order is N, CA, C.
  This is based on the output of nerf.py. Residues are of
  class Residue from the gen_data file.'''
  
  mname = loop._name
  
  pstr  =   "REMARK 950 LOOPNAME " + mname + "\n"
  pstr  +=  "REMARK 950 METHOD    AUTOGEN" + "\n"
  pstr  +=  "REMARK 950 RESOLUTION 6.000" + "\n"
  pstr  +=  "REMARK 950 R-FACTOR   0.000" + "\n"
  pstr  +=  "REMARK 950 R-FREE     0.000" + "\n"
  pstr  +=  "REMARK 950 CHAIN-TYPE LABEL ORIGINAL" + "\n"
  pstr  +=  "REMARK 950 CHAIN H    H    H" + "\n"

  idx = 0

  coords = loop._backbone
  residues = loop._residues

  assert len(coords) > 0

  for i in range(0, len(coords)):
    atom = coords[i]    
    atom_element = atom.element()
    atom_type = atom.kind

    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
    res = residues[int(math.floor(i/3))]
    line = "ATOM  "
    line += str(idx).rjust(5)
    line += atom_type.rjust(5)
    line += str(res).rjust(4)
    line += " H" 
    line += str(res._label).rjust(4)
    line += "    "
    line += str("{0:.3f}".format(atom.x)).rjust(8)
    line += str("{0:.3f}".format(atom.y)).rjust(8)
    line += str("{0:.3f}".format(atom.z)).rjust(8)
    line += str("{0:.2f}".format(1.0)).rjust(6)
    line += str("  0.00")
    line += "           " + atom_element
    line += "  "
    pstr += line + "\n" 
    idx += 1

  pstr += "TER" + str(idx).rjust(8) + "      " + str(residues[len(residues)-1]) + " H" + str(i).rjust(4)
  return pstr

if __name__ == "__main__":
  import repping
  import acids
  
  residues = [ 
        repping.Residue(acids.AminoShort.ALA, 0, 0, 0, 0, 0),
        repping.Residue(acids.AminoShort.GLY, 1, 1, 0, 0, 0),
      ]
  
  loop = repping.Loop("test")
  loop.add_residue(repping.Residue(acids.AminoShort.ALA, 0, 0, 0, 0, 0))
  loop.add_residue(repping.Residue(acids.AminoShort.GLY, 1, 1, 0, 0, 0))

  coords = [[1,1,1],[1,1,1],[1,1,1],[2,2,2],[2,2,2],[2,2,2]]
  for coord in coords:
    loop.add_backbone_atom(repping.Atom("C",coord[0],coord[1],coord[2]))

  print(printpdb(loop))
  

