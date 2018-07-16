"""
torsion_to_coord.py - Given torsion angles, create a CA PDB
author : Benjamin Blundell
email : me@benjamin.computer

Taken from Andrew Martin's program written in C,
this program takes an angle structure and prints
out a PDB of the carbon alpha backbone.

"""

import math

# Params from CHARMm

ANGLE_C     = 117.5 * math.pi / 180.0
ANGLE_N     = 120.0 * math.pi / 180.0
ANGLE_CA    = 111.6 * math.pi / 180.0
DIST_CA_C   = 1.52
DIST_C_N    = 1.33
DIST_N_CA   = 1.49
ETA         = 1.0e-7
ETA2        = ETA * ETA

class Entry():
  ''' Basic entry in the loop.'''
  def __init__(self):
    self.x = 0.0
    self.y = 0.0
    self.z = 0.0
    self.name = ""
    self.prev = None

def _make_atom(p, q, r, theta, bond, phi):
  stht = math.sin(math.pi - theta)
  ctht = math.cos(math.pi - theta)
  sphi = math.sin(phi)
  cphi = math.cos(phi)
  bsin = bond * stht

  x4 = bond * ctht
  y4 = bsin * cphi
  z4 = bsin * sphi

  x3 = r.x
  y3 = r.y
  z3 = r.z

  x1 = p.x - x3
  y1 = p.y - y3
  z1 = p.z - z3

  x2 = q.x - x3
  y2 = q.y - y3
  z2 = q.z - z3

  lxz22 = x2*x2 + z2*z2
  l2    = math.sqrt(lxz22+y2*y2)
  lxz2  = math.sqrt(lxz22)

  ovl2 = 1.0/l2
  
  if l2 < ETA:
    ovl2 = 1.0/ETA

  if lxz2 < ETA:
    xx1 = x1
    x2o = 1.0
    z2o = 0.0
  else:
    ovlxz2 = 1.0/lxz2
    x2o    = x2 * ovlxz2
    z2o    = z2 * ovlxz2
    xx1    = x1*x2o + z1*z2o
    z1     = z1*x2o - x1*z2o

  xz2o   = lxz2 * ovl2
  y2o    = y2   * ovl2

  x1     = -xx1*xz2o - y1*y2o
  y1     = xx1*y2o   - y1*xz2o

  lyz1   = math.sqrt(y1*y1 + z1*z1)
  ovlyz1 = 1.0 / lyz1

  y1o    = y1 * ovlyz1
  z1o    = z1 * ovlyz1

  yy4    = y1o*y4 - z1o*z4
  zz4    = y1o*z4 + z1o*y4
  xx4    = y2o*yy4 - xz2o*x4

  y4     = -xz2o*yy4 - y2o*x4
  x4     = x2o*xx4 - z2o*zz4
  z4     = z2o*xx4 + x2o*zz4

  return (x4 + x3, y4 + y3, z4 + z3)


def process(model_data, do_omega=False):
  ''' Go and process the torsions.'''
  # angles are in radians atm  
  # angles[mname]["angles"] = aa
  # angles[mname]["residues"] = aa

  entries = []

  start = Entry()
  start.name = "N"
  start.x = -DIST_C_N
  entries.append(start)

  p = Entry()
  p.name = "CA"
  p.prev = start
  entries.append(p)

  c = Entry()
  c.name = "C"
  c.prev = p
  c.x = DIST_CA_C * math.sin(ANGLE_CA)
  c.y = DIST_CA_C * math.cos(ANGLE_CA)
  entries.append(c)
  r = c
  rr = len(model_data["residues"])

  for i in range(0, rr):
    ma =  model_data["angles"][i]    
    phi = ma[0]
    psi = ma[1]
    omega = math.pi # Andrew's default
    if do_omega:
      omega = ma[2]

    if i != 0:
      atom = Entry()
      atom.name = "C"
      atom.x, atom.y, atom.z = _make_atom(r.prev.prev, r.prev, r, ANGLE_CA, DIST_CA_C, phi)
      entries.append(atom)
      atom.prev = r
      r = atom

    if i != rr - 1:
      atom = Entry()
      atom.name = "N"
      atom.x, atom.y, atom.z = _make_atom(r.prev.prev, r.prev, r, ANGLE_C, DIST_C_N, psi)
      entries.append(atom)
      atom.prev = r
      r = atom

      atom = Entry()
      atom.name = "CA"
      atom.x, atom.y, atom.z = _make_atom(r.prev.prev, r.prev, r, ANGLE_N, DIST_N_CA, omega)
      entries.append(atom)
      atom.prev = r
      r = atom

  return entries

def printpdb(mname, entries, residues):  
  pstr  =   "REMARK 950 LOOPNAME " + mname + "\n"
  pstr  +=  "REMARK 950 METHOD    AUTOGEN" + "\n"
  pstr  +=  "REMARK 950 RESOLUTION 6.000" + "\n"
  pstr  +=  "REMARK 950 R-FACTOR   0.000" + "\n"
  pstr  +=  "REMARK 950 R-FREE     0.000" + "\n"
  pstr  +=  "REMARK 950 CHAIN-TYPE LABEL ORIGINAL" + "\n"
  pstr  +=  "REMARK 950 CHAIN H    H    H" + "\n"

  idx = 0
  for i in range(0, len(entries)):
    atom = entries[i]
    # TODO - we could go for higher resolution than two decimal places here?
    if atom.name == "CA": 
      res = residues[int(math.floor(i/3))] 
      line = "ATOM   "
      line += str(idx).rjust(4)
      line += str("CA").rjust(4)
      line += str(res[0]).rjust(5)
      line += " H" 
      line += str(res[1]).rjust(4)
      line += str("{0:.2f}".format(atom.x)).rjust(12)
      line += str("{0:.2f}".format(atom.y)).rjust(8)
      line += str("{0:.2f}".format(atom.z)).rjust(8)
      line += "  1.00  0.00           C"
      pstr += line + "\n" 
      idx += 1

  pstr += "TER" + str(idx).rjust(8) + "      " + residues[len(residues)-1][0] + " H" + str(i).rjust(4)
  return pstr

def printpdb_real(mname, rows):  
  print("REMARK 950 LOOPNAME " + mname) 
  print("REMARK 950 METHOD    AUTOGEN")
  print("REMARK 950 RESOLUTION 6.000")
  print("REMARK 950 R-FACTOR   0.000")
  print("REMARK 950 R-FREE     0.000")
  print("REMARK 950 CHAIN-TYPE LABEL ORIGINAL")
  print("REMARK 950 CHAIN H    H    H")

  i = 0
  fidx = ""
  fres = ""

  for row in rows: 
    line = "ATOM   "
    line += str(i).rjust(4)
    line += str("CA").rjust(4)
    line += str(row[4]).rjust(5)
    fres = str(row[4])
    line += " H" 
    line += str(row[6]).rjust(4)
    fidx = str(row[6])
    line += str("{0:.2f}".format(row[8])).rjust(12)
    line += str("{0:.2f}".format(row[9])).rjust(8)
    line += str("{0:.2f}".format(row[10])).rjust(8)
    line += "  1.00  0.00           C"
    print(line)
    i += 1

  print("TER" + str(i).rjust(8) + "      " + fres + " H" + fidx.rjust(4))

if __name__ == "__main__" : 
  from gen_torsions import gen    
  import random

  angles = gen()
  mname = list(angles.keys())[random.randint(0,len(angles)-1)]
  entries = process(angles[mname])
  print(printpdb(mname, entries, angles[mname]["residues"]))

  #from gen_data import get_model_ca
  #rows = get_model_ca(mname)
  #printpdb_real(mname, rows)
