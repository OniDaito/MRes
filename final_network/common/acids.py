"""
acids.py - define our amino acid types and limits
author : Benjamin Blundell
email : me@benjamin.computer
"""

import os, math, pickle
from enum import Enum

NUM_ACIDS = 20

TRIPLES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TYR", "TRP", "VAL"]

AminoShort = Enum("AminoShort","ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TYR TRP VAL")

LETTERS = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","Y","W","V"]

# 5D compact representation - we tend to normalise this
lookup_table = {}
lookup_table["ALA"] = [0.189, -3.989, 1.989, 0.14, 1.009]
lookup_table["ARG"] = [5.007, 0.834, -2.709, -2.027, 3.696]
lookup_table["ASN"] = [7.616, 0.943, 0.101, 3.308, 0.207]
lookup_table["ASP"] = [7.781, 0.03, 1.821, 1.376, -3.442]
lookup_table["CYS"] = [-5.929, -4.837, 6.206, 2.884, 5.365]
lookup_table["GLN"] = [5.48, 1.293, -3.091, -2.348, 1.628]
lookup_table["GLU"] = [7.444, 1.005, -2.121, -1.307, -1.011]
lookup_table["GLY"] = [4.096, 0.772, 7.12, 0.211, -1.744]
lookup_table["HIS"] = [3.488, 6.754, -2.703, 4.989, 0.452]
lookup_table["ILE"] = [-7.883, -4.9, -2.23, 0.99, -2.316]
lookup_table["LEU"] = [-7.582, -3.724, -2.74, -0.736, -0.208]
lookup_table["LYS"] = [5.665, -0.166, -2.643, -2.808, 2.474]
lookup_table["MET"] = [-5.2, -2.547, -3.561, -1.73, 0.859]
lookup_table["PHE"] = [-8.681, 4.397, -0.732, 1.883, -1.987]
lookup_table["PRO"] = [4.281, -2.932, 2.319, -3.269, -4.451]
lookup_table["SER"] = [4.201, -1.948, 1.453, 1.226, 1.014]
lookup_table["THR"] = [0.774, -3.192, 0.666, 0.07, 0.407]
lookup_table["TRP"] = [-8.492, 9.958, 4.874, -5.288, 0.672]
lookup_table["TYR"] = [-6.147, 7.59, -2.065, 2.413, -0.562]
lookup_table["VAL"] = [-6.108, -5.341, -1.953, 0.025, -2.062]

def amino_to_label (ashort):
  if ashort == None:
    return ("***")
  return str(ashort.name)

def amino_pos (ashort):
  ''' Return the position/index of this acid.'''
  idx = 0
  for aa in AminoShort:
    if str(aa) == str(ashort):
      return idx
    idx+=1
  return -1

def acid_to_bitmask(ashort):
  ''' Acid to our bitmask.'''
  bitmask = []
  for i in range(0,NUM_ACIDS):
    if i == amino_pos(ashort):
      bitmask.append(1)
    else:
      bitmask.append(0)
  return bitmask

def bitmask_to_acid(mask):
  ''' bitmask back to the acid label.'''
  for i in range(0,NUM_ACIDS):
    if mask[i] == True:
      return AminoShort(i+1)
  return None

def normalise(va):
  l = math.sqrt(sum([a * a for a in va]))
  return [ a/l for a in va]

def vector_to_acid(vec):
  ''' vector back to the acid label.'''

  for k in lookup_table.keys():
    tt = 0
    vv = lookup_table[k]
    for i in range(0,5):
      tt += math.fabs(vv[i]-vec[i])

    if tt < 0.0001:
      return label_to_amino(str(k))

  # Try again but normalised
  for k in lookup_table.keys():
    tt = 0
    vv = lookup_table[k]
    vt = normalise(vv)
    for i in range(0,5):
      tt += math.fabs(vt[i]-vec[i])

    if tt < 0.0001:
      return label_to_amino(str(k))

  return None

def letter_to_amino(label):
  if label == "A":
    return AminoShort.ALA
  elif label == "R":
    return AminoShort.ARG
  elif label == "N":
    return AminoShort.ASN
  elif label == "D":
    return AminoShort.ASP
  elif label == "C":
    return AminoShort.CYS
  elif label == "E":
    return AminoShort.GLU
  elif label == "Q":
    return AminoShort.GLN
  elif label == "G":
    return AminoShort.GLY
  elif label == "H":
    return AminoShort.HIS
  elif label == "I":
    return AminoShort.ILE
  elif label == "L":
    return AminoShort.LEU
  elif label == "K":
    return AminoShort.LYS
  elif label == "M":
    return AminoShort.MET
  elif label == "F":
    return AminoShort.PHE
  elif label == "P":
    return AminoShort.PRO
  elif label == "S":
    return AminoShort.SER
  elif label == "T":
    return AminoShort.THR
  elif label == "Y":
    return AminoShort.TYR
  elif label == "W":
    return AminoShort.TRP
  elif label == "V":
    return AminoShort.VAL
  else:
    print("Error converting letter to AminoShort", label)
    return None

def amino_to_letter(aa):
  return LETTERS[amino_pos(aa)]

def label_to_amino(label):
  if label == "ALA":
    return AminoShort.ALA
  elif label == "ARG":
    return AminoShort.ARG
  elif label == "ASN":
    return AminoShort.ASN
  elif label == "ASP":
    return AminoShort.ASP
  elif label == "CYS":
    return AminoShort.CYS
  elif label == "GLU":
    return AminoShort.GLU
  elif label == "GLN":
    return AminoShort.GLN
  elif label == "GLY":
    return AminoShort.GLY
  elif label == "HIS":
    return AminoShort.HIS
  elif label == "ILE":
    return AminoShort.ILE
  elif label == "LEU":
    return AminoShort.LEU
  elif label == "LYS":
    return AminoShort.LYS
  elif label == "MET":
    return AminoShort.MET
  elif label == "PHE":
    return AminoShort.PHE
  elif label == "PRO":
    return AminoShort.PRO
  elif label == "SER":
    return AminoShort.SER
  elif label == "THR":
    return AminoShort.THR
  elif label == "TYR":
    return AminoShort.TYR
  elif label == "TRP":
    return AminoShort.TRP
  elif label == "VAL":
    return AminoShort.VAL
  else:
    print("Error converting string to AminoShort", label)
    return None

if __name__ == "__main__":
  l = AminoShort.PHE  
  mask = []

  for i in range(0,NUM_ACIDS):
    mask.append(False)

  mask[5] = True
  print(l, amino_pos(l), bitmask_to_acid(mask))
