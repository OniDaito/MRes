"""
scaffold.py - Take a pdb file, remove the CDR-H3
and graft on our own loop
author : Benjamin Blundell
email : me@benjamin.computer

We are assumming that the PDB has the Martin numbering scheme here
because of how we extract the CDR-H3

"""

import numpy as np
import math, os, sys, argparse
from Bio.PDB import *

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.geom import *
  from common.endpoint import InverseK
  from common.nerf import NeRF 

class Scaffold():
  def __init__(self):
    self.parser = PDBParser()
    self.bond_lengths = { "N_TO_A" : 1.4615,  "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
    self.bond_angles = { "A_TO_C" : math.radians(109), "C_TO_N" : math.radians(115), "N_TO_A" : math.radians(121) }

  def compute_start(self, atoms):
    # Assuming N,CA,C order
    N = atoms[0]
    CA = atoms[1]
    C = atoms[2]
    d = self._place_atom(N,CA,C,self.bond_angles["C_TO_N"], math.pi, self.bond_lengths["C_TO_N"])
    return d

  def compute_end(self, atoms):
    # TODO -  can we actually go backwards with NeRF? Lets see!
    # Assuming N,CA,C order
    N = atoms[0]
    CA = atoms[1]
    C = atoms[2]
    d = self._place_atom(C,CA,N,self.bond_angles["C_TO_N"], math.pi, self.bond_lengths["C_TO_N"])
    return d

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

  def gen_rot_mat_end(self, rstart, rend, rcoom, landing, pstart, pend, pcom, axisa):
    ''' Given the start, end positions, com and the landing, generate the matrix 
    to align with the end residue.'''
    qone = Quat()
    qone.from_to(axisa, norm(landing)) 

    # We want a rotation that deals with a poor initial placement
    # We look at the start and end points of both loops and rotate around
    # the takeoff till they match up
    # As we are rotating around bx again, we need to know which way to rotate
    
    dpe = np.matrix(qone.get_matrix()).dot(pend).A1
    dps = np.matrix(qone.get_matrix()).dot(pstart).A1
    
    proj_a = norm(sub(dpe,dps)) 
    proj_b = norm(sub(rend,rstart))
  
    proj_a = norm( sub(proj_a, mults(axisa, dot(proj_a, axisa))))
    proj_b = norm( sub(proj_b, mults(axisa, dot(proj_b, axisa))))
    rot_vec = norm(cross(proj_a, proj_b))
    rot_angle = math.acos(dot(proj_a, proj_b))
    qtwo = Quat()
    rot_angle = 0
    qtwo.from_axis_angle(rot_vec, rot_angle)

    # Concat quaternions and return matrix
    r = np.matrix(qtwo.get_qmult(qone).get_matrix())
    return r

  def gen_rot_mat_start(self, rstart, rend, rcom, takeoff, pstart, pend, pcom, axisa):
    ''' Given the start, end positions and CoM of each loop, 
    compute the rotation matrix for both takeoff'''     
    a = norm(axisa)
    b = norm(takeoff)
    
    qone = Quat()
    qone.from_to(a, b) 

    # We want a rotation that deals with a poor initial placement
    # We look at the start and end points of both loops and rotate around
    # the takeoff till they match up
    # As we are rotating around bx again, we need to know which way to rotate
    
    dpe = np.matrix(qone.get_matrix()).dot(pend).A1
    dps = np.matrix(qone.get_matrix()).dot(pstart).A1
    
    proj_a = norm(sub(dpe,dps)) 
    proj_b = norm(sub(rend,rstart))
  
    proj_a = norm( sub(proj_a, mults(takeoff, dot(proj_a, takeoff))))
    proj_b = norm( sub(proj_b, mults(takeoff, dot(proj_b, takeoff))))
    rot_vec = norm(cross(proj_a, proj_b))
    rot_angle = math.acos(dot(proj_a, proj_b))
    qtwo = Quat()
    qtwo.from_axis_angle(rot_vec,rot_angle)

    # Concat quaternions and return matrix
    r = np.matrix(qtwo.get_qmult(qone).get_matrix())
    return r

  def gen_rot_mat_com(self, rhinge, rcom, rend, phinge, pcom, pend):
    ''' Given the start, end positions and CoM of each loop, 
    compute the rotation we need from the CoM'''      
    
    a = norm(sub(pcom,phinge))
    b = norm(sub(rcom,rhinge))
    

    qone = Quat()
    qone.from_to(a, b) 
  
    # We want a rotation that deals with a poor initial placement
    # We look at the start and end points of both loops and rotate around
    # the takeoff till they match up
    # As we are rotating around bx again, we need to know which way to rotate
    
    dpe = np.matrix(qone.get_matrix()).dot(pend).A1
    dps = np.matrix(qone.get_matrix()).dot(pcom).A1
    
    proj_a = norm(sub(dpe,dps)) 
    proj_b = norm(sub(rend,rcom))
  
    proj_a = norm( sub(proj_a, mults(b, dot(proj_a, b))))
    proj_b = norm( sub(proj_b, mults(b, dot(proj_b, b))))
    rot_vec = norm(cross(proj_a, proj_b))
    rot_angle = math.acos(dot(proj_a, proj_b))
    qtwo = Quat()
    qtwo.from_axis_angle(rot_vec,rot_angle)
 
    # Concat quaternions and return matrix
    r = np.matrix(qtwo.get_qmult(qone).get_matrix())

    return r


  def process_pdb(self, pdb_path): 
    ''' Read a PDB return the start, end and matrix positions.'''
    try:
      bn = os.path.basename(pdb_path),
      st = self.parser.get_structure(bn, pdb_path)
    except:
      print("Failed to parse:", pdb_path)
      return

    models = st.get_models()
    start = []
    end = []
    rotmatrix = [1,0,0,0,1,0,0,0,1]

    for model in models:
      mname = os.path.splitext(model.get_full_id()[0][0])[0]
      print("Working with ", mname)

      try:
        # Some PDBs are missing atoms so make sure we have some :/
        # This code is a bit rough but might work
        atom_count = 0

        for atom in model.get_atoms():
          atom_count +=1
          if atom_count > 10:
            break
        
        if atom_count < 10:
          print("PDB Model contains no atoms.")
          continue
 
        plane_94 = []
        plane_106 = []
        com = (0,0,0) # Centre of mass
        nna = 0
        loop_atoms = []

        for atom in model.get_atoms():
          #import pdb; pdb.set_trace()
          #print(atom, dir(atom))
          res = atom.get_parent()
          chain = res.get_parent()
          if str(chain) == "<Chain id=H>":
            aserial = int(atom.get_serial_number())
            aname = atom.get_name()
            aresname = res.resname
            achainid = chain.id
            aresseq = int(res.id[1])
            
            if aname == "N" or aname == "C" or aname == "CA":
              ax = float(atom.get_coord()[0])
              ay = float(atom.get_coord()[1])
              az = float(atom.get_coord()[2])

              # Calculate the starting position
              if aresseq == 94:
                # Assuming the correct order of N,CA,C
                plane_94.append((ax,ay,az))

              # Go back from 103 to generate the last Carbon position
              elif aresseq == 103:
                plane_106.append((ax,ay,az))
                break
            
              elif aresseq > 94 and aresseq < 103:
                com = add(com,(ax,ay,az))
                loop_atoms.append((ax,ay,az))
                nna +=1

        start = loop_atoms[0]
        end = loop_atoms[-1]
        com = mults(com, 1.0/nna)
        takeoff = norm(sub(loop_atoms[1], loop_atoms[0]))
        landing = norm(sub(loop_atoms[-1], loop_atoms[-2]))
    
        #io = PDBIO()
        #io.set_structure(s)
        #io.save('out.pdb')

      except Exception as e:
        print ("Exception in model insert: ", e)

    return (start, end, com, takeoff, landing)     
  
  def read_angles(self, filepath, loop):
    with open(filepath, 'r') as f:
      lines = f.readlines()
      idx = 0
      for line in lines:
        tokens = line.split(",")
        phi = float(tokens[1])
        psi = float(tokens[2])
        loop._residues[idx]._phi = math.radians(phi)
        loop._residues[idx]._psi = math.radians(psi)
        loop._residues[idx]._omega = math.pi
        idx+=1

      loop._residues[0]._phi = 0
      loop._residues[-1]._psi = 0
      loop._residues[-1]._omega = 0
  
  def find_com(self, loop):
    pred_com = (0,0,0)
    nna = 0
    for atom in loop._backbone: 
      coord = atom.tuple() 
      pred_com = add(pred_com, coord)
      atom.x = coord[0]
      atom.y = coord[1]
      atom.z = coord[2]
      nna +=1

    pred_com = mults(pred_com, 1.0 / nna)
    return pred_com

  def do_kinematic(self, loop, target):
    # Target - the actual position we are after for the final Atom
    # This needs to be converted from world down to loop by inverting matrix
    # then subtracting the start position in the world of the scaffold
    # We perform an update of the angles with inverse kinematics, reconstruct
    # the loop from these angles with NeRF then write out.
    #from numpy.linalg import inv
    #invmat = inv(rotm)
    #target = np.array(end, dtype=np.float32)
    #target = sub(target, start)
    #target = invmat.dot(np.array(target)).A1 

    nrf = NeRF()
    ik = InverseK(len(loop._residues), lrate = 0.04, steps = 100, min_dist = 0.25 )
    (finalpos, results) = ik.train(loop._residues, target, record = True)

    idr = 0
    for res in loop._residues:
      res._phi = results[-1][idr]
      res._psi = results[-1][idr+1]
      idr += 2

    nrf.gen_loop_coords(loop)

  def apply(self, scaffold, loop, kinematic = False):
    ''' Modify the incoming loop by the start, end and com position.'''
    # We have a scaffold so we can move our loop around for grafting
    # Find the loop's start, end and com, to generate our rot matrix
    start, end, com, takeoff, landing = self.process_pdb(scaffold) 
    pred_com = self.find_com(loop)

    phinge = mults(add(loop._backbone[-1].tuple(), loop._backbone[0].tuple()),0.5)
    rhinge = mults(add(start,end),0.5) 
    rotc = self.gen_rot_mat_com(rhinge, com, end, phinge, pred_com, loop._backbone[-1].tuple())

    #rotm = self.gen_rot_mat_start(start, end, com, takeoff, loop._backbone[0].tuple(), loop._backbone[-1].tuple(), pred_com, (0,1,0)) 
  
    #baxis = norm(sub(loop._backbone[-1].tuple(), loop._backbone[-2].tuple()))
    #rotn = self.gen_rot_mat_end(start, end, com, landing, loop._backbone[0].tuple(), loop._backbone[-1].tuple(), pred_com, baxis)

    build_start = False
    new_coords = []

    for atom in loop._backbone:
      coord = atom.tuple()
      nc = np.array(coord)
      nc = sub(nc,phinge)
      nc = rotc.dot(nc).A1  
      nc = add(rhinge, nc)
      new_coords.append(nc)

    # We now have the final world coordinates.
    # We can stop here or perform kinematics

    if kinematic:
      #for atom in loop._backbone:
      #  print(atom.tuple())

      print(new_coords[-1])
      target = sub(end,new_coords[-1])
      from numpy.linalg import inv
      invmat = inv(rotc)
      target = np.array(target, dtype=np.float32)
      target = invmat.dot(np.array(target)).A1 
      target = add(loop._backbone[-1].tuple(),target)
      print("length", length(sub(end,new_coords[-1])))		
      self.do_kinematic(loop, target)
  
      #target = sub(end,new_coords[-1])
      new_coords[:] = []
      
      # redo the phinge and rotc etc
      #phinge = mults(add(loop._backbone[-1].tuple(), loop._backbone[0].tuple()),0.5)
      #pred_com = self.find_com(loop)
      #rotc = self.gen_rot_mat_com(rhinge, com, phinge, pred_com)


      for atom in loop._backbone:
        coord = atom.tuple()
        nc = np.array(coord)
        nc = sub(nc,phinge)
        nc = rotc.dot(nc).A1  
        coord = add(rhinge, nc) 
        #coord = add(coord,target)
        new_coords.append(coord)
    
    idx = 0
    for atom in loop._backbone:
      coord = new_coords[idx]
      atom.x = coord[0]
      atom.y = coord[1]
      atom.z = coord[2]
      idx += 1

    #print(results)
    #from copy import deepcopy
    #num_frames = len(results)
    #frames = []

    #for i in range(0, num_frames):
    #  frame = deepcopy(residues)
    #  for res in range(0, len(frame)):
    #    residue = frame[res]
    #    residue._phi = results[i][res*2]
    #    residue._psi = results[i][res*2+1]
    #    residue._omega = math.pi
    #  frames.append(frame)
       
    #from anim_json import *
    #to_json_animation(name, frames)
    #to_json_target(name, residues)


if __name__ == "__main__":
  from geom import *
  from endpoint import InverseK
  from nerf import NeRF 

  parser = argparse.ArgumentParser(description="Process arguments")
  parser.add_argument('scaffold', metavar='scaffold', help='The PDB to use as the scaffold')
  parser.add_argument('loop', metavar='loop', help='The PDB to use as the loop')
  parser.add_argument('angles', metavar='loop', help='The loop phi/psi angles')

  args = vars(parser.parse_args())
  p = Scaffold()
  from pdb import loop_from_pdb, loop_to_pdb
  loop = loop_from_pdb(args["loop"])
  p.read_angles(args["angles"], loop)
  p.apply(args["scaffold"], loop, kinematic=True)
  loop_to_pdb("neuralknit.pdb", loop)

