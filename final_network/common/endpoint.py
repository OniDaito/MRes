"""
endpoint.py - The NeRF algorithm as applied to Inverse Kinematics
author : Benjamin Blundell
email : me@benjamin.computer

This is the tensorflow version of the NeRF program. In this way
we can perform gradient descent to refine the position. This version
only tries to match the endpoint.

"""

import tensorflow as tf
import numpy as np
import math, itertools, os

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  from common.geom import *

class InverseK(object):

  def __init__(self, cdr_length, lrate = 0.05, steps = 100, min_dist = 0.5 ):
    ''' To initialise, just pass in the length of this loop.'''
    # Data constants for amino acid backbones
    self.bond_lengths = { "N_TO_A" : 1.4615, "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
    self.bond_angles = { "A_TO_C" : math.radians(109), "C_TO_N" : math.radians(115), "N_TO_A" : math.radians(121) }
    self.bond_order = ["C_TO_N", "N_TO_A", "A_TO_C"]
    # bond_angles and lengths
    key = "C_TO_N"
    angle = self.bond_angles[key]
    length = self.bond_lengths[key]
    self._angles = []
    self._lengths = []
    self._angles.append(angle)
    self._lengths.append(length)
    self.cdr_length = cdr_length
    self.num_steps = steps
    self.learning_rate = lrate
    self.min_dist = min_dist # Distance in angstroms - we stop early when we get below this

    for i in range(0, self.cdr_length * 3):
      (angle, length, key) = self.next_data(key)
      self._angles.append(angle)
      self._lengths.append(length)

    self._lengths = np.array(self._lengths, dtype=np.float32)
    self._angles = np.array(self._angles, dtype=np.float32)
      
    # Initial positions
    self._initial_positions = np.array([ [0, -1.355, 0], [0, 0, 0], [1.4466, 0.4981, 0] ], dtype=np.float32)

  def next_data(self, key):
    ''' Loop over our bond_angles and bond_lengths '''
    ff = itertools.cycle(self.bond_order)
    for item in ff:
      if item == key:
        next_key = next(ff)
        break
    return (self.bond_angles[next_key], self.bond_lengths[next_key], next_key)

  def normalise(self, x):
    ''' Normalise a vector '''
    return x / tf.sqrt(tf.reduce_sum(tf.square(x)))

  def place_atoms(self, torsion_angles) :
    ''' Place all of our atoms. Based on place_atom but does the entire cdr_length. '''
    positions = self._initial_positions
   
    for i in range(0, self.cdr_length - 1): # -1 as we don't place with the last psi
      idy = i * 2 + 1 # move along one to start on a psi as we should 
      for j in range(0, 3):
        idx = i * 3 + j
        if j == 1:
          d = self._place_atom(positions, self._angles[idx], math.pi, self._lengths[idx])
        else:
          d = self._place_atom(positions, self._angles[idx], torsion_angles[idy], self._lengths[idx])
          idy += 1

        positions = tf.stack([positions[1], positions[2], d], 0)

    # Return the last atom we are placing, the C 
    return positions[-1] 

  def _place_atom (self, positions, bond_angle, torsion_angle, bond_length):
    ''' Given the three previous atoms, the required angles and the bond
    lengths, place the next atom. Angles are in radians, lengths in angstroms.''' 
    ab = tf.subtract(positions[1], positions[0])
    bc = tf.subtract(positions[2], positions[1])
    bcn = self.normalise(bc)
    R = bond_length 
    
    dx = -R * tf.cos(bond_angle)
    dy = R * tf.cos(torsion_angle) * tf.sin(bond_angle)
    dz = R * tf.sin(torsion_angle) * tf.sin(bond_angle)

    d = tf.stack([dx,dy,dz], 0) 
    n = tf.cross(ab,bcn)
    n = self.normalise(n)
    nbc = tf.cross(n,bcn)
    m = tf.stack([bcn,nbc,n], 0)
    d = tf.reduce_sum(tf.multiply(tf.expand_dims(d,-1), m), axis=0)
    d = d + positions[2]
    return d

  def basic_error(self, target, torsion_angles):
    """ Our basic error function. Reduce the difference in positions."""
    d = self.place_atoms(torsion_angles)
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(target, d),2)))

  def get_torsions(self, residues):
    ''' Given residues, grab an array of angles in radians.'''
    torsions = []
    for residue in residues:
      torsions.append(residue._phi)
      torsions.append(residue._psi)

    return np.array(torsions, dtype=np.float32)

  def dist(self, pos, target):
    xd = pos[0] - target[0]
    yd = pos[1] - target[1]
    zd = pos[2] - target[2]
    dd = xd * xd + yd * yd + zd * zd
    return math.sqrt(dd)

  def train (self, residues, target, record = False) : 
    ''' Perform the training. Assume that the loop starts at 0,0,0 - move the target to 
    reflect the fact that we start at -1.3 on Y axis actually, due to NeRF.'''
    target = add(target, [0, -1.355, 0]) 

    # Start the tensorflow section
    torsions = self.get_torsions(residues)
    results = []
    with tf.Session() as sess:
      place_torsion = tf.placeholder("float", None)
      place_target = tf.placeholder("float", None)
      place_initx = tf.placeholder("float", None)
      # The variable that stands in for the real torsions. We will optimise this
      x = tf.Variable(torsions, "torsion")

      # Assign the torsions to variable x
      error = self.basic_error(place_target, x)
      train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(error) 
      #train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(error)
      tf.global_variables_initializer().run() 
      
      # Actually perform the gradient descent
      for stepnum in range(0, self.num_steps):
        sess.run([train_step], feed_dict={ place_torsion: torsions, place_target: target})
        #tx = x.eval()
        #er = sess.run(error, feed_dict = {place_target: target})
        #print("***")
        #print("ER:",er)
        #for i in range(0, len(tx)):
        #  diff = math.fabs(tx[i] - torsions[i])
        #  print(diff)


        # Print out the actual position and torsion angles for the last carbon alpha
        if stepnum % 4 == 0:
          pa = self.place_atoms(x)
          cpos = pa.eval(feed_dict={ place_target: target, place_torsion: torsions})
          result = sess.run(x, feed_dict={ place_torsion: torsions, place_target: target}) 
          print(stepnum, target, cpos)
          if record:
            results.append(result)
          if self.dist(cpos, target) <= self.min_dist:
            break
        #print("cpos:", cpos) 
        #print("---\n")
      # Final angles
      pa = self.place_atoms(x)
      cpos = pa.eval(feed_dict={place_torsion: torsions, place_target: target})
      finalres = sess.run(x, feed_dict={place_torsion: torsions, place_target: target}) 
  
      if record:
        results.append(finalres)

    sess.close()
    return (cpos, results) 

if __name__ == "__main__":

  # Test Data obtained from 3NH7_1 
  test_cdr_length = 11
  # For now, assume we have the correct omegas for the prolines. It just makes it easy
  # Torsions are all arranged here in the correct order (prev psi, prev omega, phi)

  torsions = np.array([0.0, 142.95, 173.209, 
    -147.449, 138.084, -176.98,
    -110.138, 138.08, 162.29,
    -101.068, -96.169, 167.885,
    -78.779, -44.373, 175.878,
    -136.836, 164.182, -172.224,
    -63.91, 143.817, 168.896, 
    -144.503, 158.705, 175.872,
    -96.842, 103.724, -172.34,
    -85.734, -18.138, -172.979
    -150.084, 0.0, 0.0
    ])

  torsions = np.array([0.0, 142.95, 
    -147.449, 138.084,
    -110.138, 138.08,
    -101.068, -96.169,
    -78.779, -44.373,
    -136.836, 164.182,
    -63.91, 143.817,
    -144.503, 158.705,
    -96.842, 103.724,
    -85.734, -18.138,
    -150.084, 0.0
    ])
 
  import repping, acids
  from geom import *
  
  name = "3NH7_1"
  residues = []
  
  torsions = np.array(list(map(math.radians, torsions)), dtype=np.float32)
  
  residues.append(repping.Residue(acids.AminoShort.GLU, 0, "0", torsions[0], torsions[1], math.radians(173.209)))
  residues.append(repping.Residue(acids.AminoShort.ARG, 1, "1", torsions[2], torsions[3], math.radians(-176.98)))
  residues.append(repping.Residue(acids.AminoShort.TRP, 2, "2", torsions[4], torsions[5], math.radians(162.29)))
  residues.append(repping.Residue(acids.AminoShort.HIS, 3, "3", torsions[6], torsions[7], math.radians(167.885)))
  residues.append(repping.Residue(acids.AminoShort.VAL, 4, "4", torsions[8], torsions[9], math.radians(175.878)))
  residues.append(repping.Residue(acids.AminoShort.ARG, 5, "5", torsions[10], torsions[11], math.radians(-172.224)))
  residues.append(repping.Residue(acids.AminoShort.GLY, 6, "6", torsions[12], torsions[13], math.radians(168.896)))
  residues.append(repping.Residue(acids.AminoShort.TYR, 7, "7", torsions[14], torsions[15], math.radians(175.872)))
  residues.append(repping.Residue(acids.AminoShort.PHE, 8, "8", torsions[16], torsions[17], math.radians(172.34)))
  residues.append(repping.Residue(acids.AminoShort.ASP, 9, "9", torsions[18], torsions[19], math.radians(-172.979)))
  residues.append(repping.Residue(acids.AminoShort.HIS, 10, "10", torsions[20], torsions[21], 0))
 
  # Target - the actual position we are after for the final Carbon Alpha
  target = np.array([-0.7506869975369328, -3.2323655926888777, -6.703822569392947 ], dtype=np.float32) 

  ik = InverseK(test_cdr_length)
  (finalpos, results) = ik.train(residues, target, record = True)

  from copy import deepcopy
  num_frames = len(results)
  frames = []

  for i in range(0, num_frames):
    frame = deepcopy(residues)
    for res in range(0, len(frame)):
      residue = frame[res]
      residue._phi = results[i][res*2]
      residue._psi = results[i][res*2+1]
      residue._omega = math.pi
    frames.append(frame)
     
  from anim_json import *
  to_json_animation(name, frames)
  to_json_target(name, residues)

