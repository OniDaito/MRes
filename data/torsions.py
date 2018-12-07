"""
torsions.py - module for working out backbone torsions
author : Benjamin Blundell
email : me@benjamin.computer

This program will derive a set of phi, psi and omega
angles when given a set of residues. Useful in 
determining protein backbones.

"""

import math

# The following are simple math functions - speeding these
# up might be a good place to start with improvements. For
# now, this is a little easier than numpy or other functions
# but eventually, we should use something faster.

def cross(u,v):
  x = (u[1]*v[2]) - (u[2]*v[1])
  y = (u[2]*v[0]) - (u[0]*v[2])
  z = (u[0]*v[1]) - (u[1]*v[0])
  return (x,y,z)

def sub(u,v):
  return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

def norm(u):
  l = 1.0 / length(u)
  return (u[0] *l, u[1] * l, u[2] * l)

def dot(u,v):
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def length(u) :
  return math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

def res_atom (residue, label):
  for rr in residue:
    if rr[0] == label:
      return (rr[3], rr[4], rr[5])

# residue is as follows:
# atom.label, atom.resseq, atom.resnum, x, y, z

def derive_angles(residues):
  """ Given the residues, lets take a look at the atoms within and dervive the angles. """
  angles = []
  Cap = (0,0,0)
  C = (0,0,0)
  Nn = (0,0,0)
  Ca = (0,0,0)
  N = (0,0,0)
  idx = 0

  for idx in range(0,len(residues)):
  
    phi = psi = omega = 0
    res = residues[idx]

    N  = res_atom(res,'N')
 
    if idx != 0:
      Cp = C
      C = res_atom(res,'C')
      Cap = Ca 
      Ca = res_atom(res,'CA')
      # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
      # ORTHONORMAL FRAME (changing basis basically)
      # Results dont seem as close as cos-1 version but apparently, this is more accurate :/ 
      # Phi
      a = sub(Cp,N)
      b = sub(Ca,N)
      d = b 
      n0a = cross(a, b)
      n1a = cross(sub(N,Ca), sub(C,Ca))
      cosphi = -dot(n0a,n1a) / (length(n0a) * length(n1a))
      nx = dot(cross(n0a,n1a),d)
      phi = math.degrees(math.acos(cosphi))
      if nx < 0: 
        phi = -180 + phi
      else :
        phi = 180 - phi

    if idx < len(residues) - 1: 
      Ca = res_atom(res,'CA')
      resn = residues[idx+1]
      Nn = res_atom(resn,'N')
      Can = res_atom(resn,'CA')
      C = res_atom(res,'C')

      # Omega - needed to get Phi and Psi correct because it might flip things around
      # we need to adjust the next angle
      d  = sub(Nn,C)
      n0c = cross(sub(Ca,C), sub(Nn,C))
      n1c = cross(sub(Can,Nn), sub(C,Nn))
      cosomega = -dot(n0c,n1c) / (length(n0c) * length(n1c)) 
      nx = dot(cross(n0c,n1c),d)
      omega = math.degrees(math.acos(cosomega))
      if nx > 0:
        omega = -omega # this seems too simple :/

      # This method uses arctan and is more reliable apparently
      #m = cross(b,n0a)
      #x = dot(n0a,n1a)
      #y = dot(m,n1a)
      #phi = math.degrees(math.atan2(y,x))

      # Psi
      a = sub(N,Ca)
      b = sub(C,Ca)
      d = b
      n0b = cross(a, b)
      n1b = cross(sub(Ca,C), sub(Nn,C))
      # Same arctan method
      #m = cross(b,n0b)
      #x = dot(n0b,n1b)
      #y = dot(m,n1b) 
      #psi = math.degrees(math.atan2(y,x))
      cospsi = -dot(n0b,n1b) / (length(n0b) * length(n1b))
      nx = dot(cross(n0b,n1b),d)
      psi = math.degrees(math.acos(cospsi))
      if nx < 0:
        psi = -180 + psi
      else :
        psi = 180 - psi

    angles.append((phi, psi, omega))
    idx += 1

  return angles
