"""
nerf.py - The NeRF algorithm
author : Benjamin Blundell
email : me@benjamin.computer

This program converts torsion angles to cartesian co-ordinates
for amino-acid back-bones. Based on the following resources:

http://onlinelibrary.wiley.com/doi/10.1002/jcc.20237/abstract
https://www.ncbi.nlm.nih.gov/pubmed/8515464
https://www.google.com/patents/WO2002073193A1?cl=en

"""

import numpy as np
import math, itertools, os

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.repping import *
  from common.geom import sub

class NeRF(object):

  def __init__(self):
    # TODO - PROLINE has different lengths which we should take into account
    # TODO - A_TO_C angle differs by +/- 5 degrees
    #bond_lengths = { "N_TO_A" : 1.4615, "PRO_N_TO_A" : 1.353, "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
    self.bond_lengths = { "N_TO_A" : 1.4615,  "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
    self.bond_angles = { "A_TO_C" : math.radians(109), "C_TO_N" : math.radians(115), "N_TO_A" : math.radians(121) }
    self.bond_order = ["C_TO_N", "N_TO_A", "A_TO_C"]

  def _next_data(self, key):
    ''' Loop over our bond_angles and bond_lengths '''
    ff = itertools.cycle(self.bond_order)
    for item in ff:
      if item == key:
        next_key = next(ff)
        break
    return (self.bond_angles[next_key], self.bond_lengths[next_key], next_key)

  def _place_atom(self, atom_a, atom_b, atom_c, bond_angle, torsion_angle, bond_length) :
    ''' Given the three previous atoms, the required angles and the bond
    lengths, place the next atom. Angles are in radians, lengths in angstroms.''' 
    # TODO - convert to sn-NeRF
    ab = np.subtract(atom_b, atom_a)
    bc = np.subtract(atom_c, atom_b)
    bcn = bc / np.linalg.norm(bc)
    R = bond_length

    # numpy is row major
    d = np.array([-R * math.cos(bond_angle),
        R * math.cos(torsion_angle) * math.sin(bond_angle),
        R * math.sin(torsion_angle) * math.sin(bond_angle)])

    n = np.cross(ab,bcn)
    n = n / np.linalg.norm(n)
    nbc = np.cross(n,bcn)

    m = np.array([ 
      [bcn[0],nbc[0],n[0]],
      [bcn[1],nbc[1],n[1]],
      [bcn[2],nbc[2],n[2]]])

    d = m.dot(d)
    d = d + atom_c
    return d
  
  def _compute_positions(self, loop, torsions):
    ''' Call this function with a set of torsions (including omega) in degrees.'''
    atoms = [Atom("N", 0, -1.355, 0), Atom("CA", 0, 0, 0), Atom("C", 1.4466, 0.4981, 0)]
    torsions = list(map(math.radians, torsions))
    key = "C_TO_N"
    angle = self.bond_angles[key]
    length = self.bond_lengths[key]

    for torsion in torsions:
      name = "N"
      if key == "N_TO_A":
        name = "CA"
      elif key == "A_TO_C":
        name = "C"

      coord = self._place_atom(atoms[-3].tuple(), atoms[-2].tuple(), atoms[-1].tuple(), angle, torsion, length)
      atoms.append( Atom(name, coord[0], coord[1], coord[2]))
      (angle, length, key) = self._next_data(key)

    return atoms

  def gen_loop_coords(self, loop, omega=False):
    ''' Given our loop, generate the coordinates for it.'''
    torsions = [] 
    
    for r in loop._residues:
      torsions.append(r.phid())
      torsions.append(r.psid())
      if omega:
        torsions.append(r.omegad())
      else:
        torsions.append(180)
  
    # Remove first, and last two
    torsions = torsions[1:]
    torsions.pop()
    torsions.pop()

    atoms = self._compute_positions(loop, torsions)
    # Adjust so our first atom is at the origin 
    offset = atoms[0].tuple()
    for atom in atoms:
      coord = sub(atom.tuple(), offset)
      atom.x = coord[0]
      atom.y = coord[1]
      atom.z = coord[2]
    loop._backbone = atoms

if __name__ == "__main__":

  from repping import *
  from acids import *

  nerf = NeRF()

  print ("3NH7_1 - using real omega")
  torsions = [ 0, 142.951668191667, 173.2,
  -147.449854444109, 137.593755455898, -176.98,
  -110.137784727015, 138.084240732612, 162.28,
  -101.068226849313, -96.1690297398444, 167.88,
  -78.7796836206707, -44.3733790929788, 175.88,
  -136.836113196726, 164.182984866024, -172.22,
  -63.909882696529, 143.817250526837, 168.89,
  -144.50345668635, 158.70503596547, 175.87,
  -96.842536650294, 103.724939588454, -172.34,
  -85.7345901579845, -18.1379473766538, -172.98,
  -150.084356709565, 0 ,0]

  residues = [ Residue(AminoShort.GLY, 0, 0, torsions[0], torsions[1], torsions[2]),
      Residue(AminoShort.GLY, 1, 1, torsions[3], torsions[4], torsions[5]),
      Residue(AminoShort.GLY, 2, 2, torsions[6], torsions[7], torsions[8]),
      Residue(AminoShort.GLY, 3, 3, torsions[9], torsions[10], torsions[11]),
      Residue(AminoShort.GLY, 4, 4, torsions[12], torsions[13], torsions[14]),
      Residue(AminoShort.GLY, 5, 5, torsions[15], torsions[16], torsions[17]),
      Residue(AminoShort.GLY, 6, 6, torsions[18], torsions[19], torsions[20]),
      Residue(AminoShort.GLY, 7, 7, torsions[21], torsions[22], torsions[23]),
      Residue(AminoShort.GLY, 8, 8, torsions[24], torsions[25], torsions[26]),
      Residue(AminoShort.GLY, 9, 9, torsions[27], torsions[28], torsions[29]),
      Residue(AminoShort.GLY, 10, 10, torsions[30], torsions[31], torsions[32])
    ]
  
  loop = Loop("3NH7_1")
  loop._residues = residues


  nerf.gen_loop_coords(loop)
  atoms0 = loop._backbone
  print(len(atoms0))
  for atom in atoms0:
    print(atom)

