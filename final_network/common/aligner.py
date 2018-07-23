"""
aligner.py - Align two pdbs and write out
author : Benjamin Blundell
email : me@benjamin.computer

This relies on the program pdbfit being in the path

"""

import os, sys, signal, subprocess

def align(pathtemplate, pathtarget):
  ''' Perform an alignment, and overwrite the target.'''
  pro = subprocess.run(["pdbfit", "-w", pathtemplate, pathtarget], stdout=subprocess.PIPE)
  result = pro.stdout.decode()
  with open(pathtarget,'w') as f:
    f.write(result)

if __name__ == "__main__":
  path = sys.argv[1]
  pdb_files = []
  for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
      pdb_extentions = ["pdb","PDB"]
      if any(x in filename for x in pdb_extentions) and "all" not in filename:
        pdb_files.append(os.path.join(os.getcwd(),os.path.join(dirname, filename)))
 
  pdb_files.sort()
  pdb_pairs = []
  
  # pair up for pred and real
  for i in range(0,len(pdb_files)-1,2):
    pdb_pairs.append((pdb_files[i], pdb_files[i+1]))

  for pair in pdb_pairs:
    align(pair[1], pair[0])



