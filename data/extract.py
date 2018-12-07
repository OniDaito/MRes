#!/usr/bin/python
# USAGE - ./extract.sh <path to summary file> <path to PDBs> <path to output>
# We ignore the first 3 lines

import fileinput
import argparse
import subprocess

if __name__ == "__main__":
  parser = argparse.ArgumentParser(\
      description='Extract Loops with pdbgetzone')

  parser.add_argument('summary', metavar='S',\
      help='Path to the summary file / db')

  parser.add_argument('pdbdir', metavar='P',\
      help='Path to the directory of PDBs')

  parser.add_argument('outdir', metavar='O',\
      help='Path to the output directory')

  args = parser.parse_args()
  counter = 0
  
  for line in fileinput.input(args.summary):
    if counter > 2:
      tokens = line.split(" ")
      cmd = ['pdbgetzone', tokens[1], tokens[2],\
        args.pdbdir + "/pdb" + tokens[0] + ".ent",\
        args.outdir + "/" + tokens[0] + "_" + tokens[1] + \
        tokens[2] + ".pdb"]
      print(cmd)
      mode = subprocess.run(cmd, capture_output=True, check=True)
      print(counter, tokens, str(mode))

    counter = counter + 1
