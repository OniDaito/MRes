"""
repping.py - The representations of our loops and residues
author : Benjamin Blundell
email : me@benjamin.computer

"""

import math, os
import numpy as np

# Filthy hack but neater than the alternative
if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  import common.acids as acids

class Residue(object):
  ''' An individual residue with angles and name associated. '''
  def __init__(self, name, index, label, phi, psi, omega):
    self._name = name
    if "AminoShort" not in str(type(name)):
      print("Error in creating residue. Name should be an AminoShort")
    self._index = index
    self._label = label
    # Angles as radians please! -_-
    self._phi = phi
    self._psi = psi
    self._omega = omega
  
  def get_index(self) :
    return self._index

  def __str__(self):
    return acids.amino_to_label(self._name)

  def phid(self): return math.degrees(self._phi)
  
  def psid(self): return math.degrees(self._psi)
  
  def omegad(self): return math.degrees(self._omega)

class Atom(object):
  ''' A class to represent our backbone atoms.'''
  def __init__(self, type_label, x, y, z):
    self.kind = type_label # N,CA,C
    self.x = x
    self.y = y
    self.z = z

  def element(self):
    if self.kind == "C" or self.kind == "CA":
      return "C"
    elif self.kind == "N":
      return "N"
    return "U"
  
  def tuple(self):
    return np.array([self.x, self.y, self.z], dtype=np.float32)

  def __str__(self):
    return self.kind + " (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

class Loop(object):
  ''' A class that represents a single CDR Loop. '''
  def __init__(self, name):
    self._name = name
    self._residues = []
    self._endpoint_distance = 0.0
    self._endpoint = [0,0,0] # x,y,z of the required endpoint
    self._backbone = []

  def add_residue(self, residue):
    if type(residue) == Residue:
      self._residues.append(residue)
      # Ensure the correct order
      self._residues = sorted(self._residues, key=Residue.get_index)
 
  def add_atom(self, atom):
    ''' add a backbone atom, in order.'''
    self._backbone.append(atom)

  def get_residues(self) : return self._residues

  def to_string(self):
    sr = ""
    for res in self._residues: 
      sr += acids.amino_to_letter(res._name)
    return sr
  
  def __str__(self):
    return self._name

  def print(self):
    print(self._name)
    print("-----")
    for res in self._residues:
      print(str(res) + ": " + str(res.phid()) + ", " + str(res.psid()) + ", " + str(res.omegad()))

    print("")


