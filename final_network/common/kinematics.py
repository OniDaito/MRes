"""
kinematics.py - The NeRF algorithm as applied to Inverse Kinematics
author : Benjamin Blundell
email : me@benjamin.computer

This is the tensorflow version of the NeRF program. In this way
we can perform gradient descent to refine the position. We attempt
to match ALL the angles rather than just the endpoint here.
"""
import tensorflow as tf
import numpy as np
import math, itertools, os

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)

class InverseK (object):

  def __init__(self, max_cdr_length, batch_size, width):
    ''' Init the class. Pass in the max_cdr_length, the batch_size and the width of the input angles,
    4 if its phi/psi and 6 if its phi/psi/omega.'''
    # Data constants for amino acid backbones
    self.bond_lengths = { "N_TO_A" : 1.4615, "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
    self.bond_angles = { "A_TO_C" : math.radians(109), "C_TO_N" : math.radians(115), "N_TO_A" : math.radians(121) }
    self.bond_order = ["C_TO_N", "N_TO_A", "A_TO_C"]

    # Setup for the tensorflow bits
    # bond_angles and lengths
    key = "C_TO_N"
    angle = self.bond_angles[key]
    length = self.bond_lengths[key]
    self._angles = []
    self._lengths = []
    self.width = width
    self._angles.append(angle)
    self._lengths.append(length)
    self.max_cdr_length = max_cdr_length
    self.batch_size = batch_size
    self.num_steps = 100
    self.learning_rate = 0.45

    for i in range(0, max_cdr_length * 3):
      (angle, length, key) = self._next_data(key)
      self._angles.append(angle)
      self._lengths.append(length)

    # Bond length and bond angles running to max_cdr_length
    self._lengths = np.array(self._lengths, dtype=np.float32)
    self._angles = np.array(self._angles, dtype=np.float32)

    # Initial positions
    self._initial_positions = np.array([ [0, -1.355, 0], [0, 0, 0], [1.4466, 0.4981, 0] ], dtype=np.float32)

  def _next_data(self, key):
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

  def _place_atoms(self, torsion_angles) :
    ''' Place all of our atoms. Based on place_atom but does the entire cdr_length. '''
    positions = self._initial_positions
    returnvals = list(self._initial_positions)

    for i in range(0, self.max_cdr_length - 1):
      idy = i * 2
      for j in range(0, 3):
        idx = i * 3 + j
        if j == 1:
          d = self._place_atom(positions, self._angles[idx], math.pi, self._lengths[idx])
        else:
          angle = torsion_angles[idy]
          d = self._place_atom(positions, self._angles[idx], angle, self._lengths[idx])
          idy += 1

        positions = tf.stack([positions[1], positions[2], d], 0)
        returnvals.append(d)

    return tf.stack(returnvals, 0)

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

  def gen_structure (self, torsion_angles) :
    ''' Format for torsion_angles should be max_cdr_length * 4. We reshape and perform
    the atan2 to get the real angles. We then recombine into cdr_length * 2 and pass 
    to place_atoms. This version doesn't do omega unlike nn25.'''
    # Reshape so the sin/cos is properly grouped
    w = int(math.floor(self.width / 2))
    t_r = tf.reshape(torsion_angles, [self.max_cdr_length, w, 2])

    # We have the sin and cos but we just want the angle so we split and combine
    tp = tf.transpose(t_r)
    ta = tp[0]
    tb = tp[1]
    torsions = tf.atan2(ta,tb)
    f_r = tf.transpose(torsions)
    f_r = tf.reshape(f_r,[self.max_cdr_length * w])
    d = self._place_atoms(f_r)
    return d

  def cost (self, gen_batch, target_batch, cdr_lengths) :
    """ Deal with a batch of stuff instead of just one. We solve for the entire length, but only  """    
    cost = 0
    for idx in range(0, self.batch_size):
      gen = tf.slice(gen_batch, [idx,0], [1,-1])
      trg = tf.slice(target_batch, [idx,0], [1,-1])
    
      ta = self.gen_structure(gen)
      tb = self.gen_structure(trg)
   
      ta = ta * self.create_mask_output(cdr_lengths[idx])
      tb = tb * self.create_mask_output(cdr_lengths[idx])

      cost += tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(ta, tb), 2)))
    
    return cost / self.batch_size

  def create_mask_input(self, length):
    ''' create a mask for our input, which
    is a [1] shape that is max_cdr * 4 long.'''
    mask = []
    for i in range(0,self.max_cdr_length * self.width):
      tt = 1
      if i >= length * self.width:
        tt = 0
      for i in range(0,self.width):
        mask.append(tt)
    return np.array(mask,dtype=np.float32)

  def create_mask_output(self, length):
    ''' create a mask for our output, which
    is a [1] shape that is max_cdr * 3 * 3 long.'''
    mask = []
    for i in range(0,self.max_cdr_length * 3):
      mm = []
      tt = 1
      if i >= length * 3:
        tt = 0
      for i in range(0,3):
        mm.append(tt)
      mask.append(mm)
    return np.array(mask,dtype=np.float32)

  def train(self, target_angles, cdr_lengths, record=False):
    with tf.Session() as sess:
      x = tf.Variable(tf.zeros([ 2, self.max_cdr_length * self.width ]), "torsions")
      place_target = tf.placeholder("float", None)
      error = self.cost(x, place_target, cdr_lengths) 
      results = []

      # TODO - Looks like we HAVE to have a gradient cap. Not sure why but it will explode
      # otherwise
      optimizer = tf.train.AdagradOptimizer(self.learning_rate)  
      gvs = optimizer.compute_gradients(error)
      capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      train_step = optimizer.apply_gradients(capped_gvs) 
      tf.global_variables_initializer().run()

      # Actual training with just gradient descent on X
      for stepnum in range(0,self.num_steps):
        sess.run([train_step], feed_dict={place_target: target_angles})
        #print(x.eval())
        if record:
          if stepnum % 5 == 0:
            #train_accuracy = error.eval( feed_dict={place_target: target_angles})   
            #print('step %d, training accuracy %g' % (stepnum, train_accuracy))
            result = sess.run(x, feed_dict={place_target: target_angles})
            results.append(result)
 
      targets = []
      for i in range(0,self.batch_size):
        pa = self.gen_structure(x[i])
        targets.append(pa.eval())

    return (targets, results)

if __name__ == "__main__" :
  ''' A test function mostly to see if this function will work at all! '''
  import os, sys
  import repping, acids

  # Test data from 3NH7_1 - phi and psi angles
  target_base = [(0,142.951668191667),
    (-147.449854444109,137.593755455898),
    (-110.137784727015,138.084240732612),
    (-101.068226849313,-96.1690297398444),
    (-78.7796836206707,-44.3733790929788),
    (-136.836113196726,164.182984866024),
    (-63.909882696529,143.817250526837),
    (-144.50345668635,158.70503596547),
    (-96.842536650294,103.724939588454),
    (-85.7345901579845,-18.1379473766538),
    (-150.084356709565,0)]

  # Test data from 3NH7_1 - phi, psi and omega angles
  #target_base = [(0,142.951668191667, 173.2),
  #  (-147.449854444109,137.593755455898, -176.98 ),
  #  (-110.137784727015,138.084240732612, 162.28),
  #  (-101.068226849313,-96.1690297398444, 167.88),
  #  (-78.7796836206707,-44.3733790929788, 175.88),
  #  (-136.836113196726,164.182984866024, -172.22),
  #  (-63.909882696529,143.817250526837, 168.89),
  #  (-144.50345668635,158.70503596547, 175.87),
  #  (-96.842536650294,103.724939588454, -172.34),
  #  (-85.7345901579845,-18.1379473766538, -172.98),
  #  (-150.084356709565,0,0)]

  # Positions worked out using python NeRF and full omega
  actual_positions = [
    [0, -1.355, 0],
    [0, 0, 0],
    [1.4466, 0.4981, 0],
    [ 1.66402739, 1.58662823, -0.72350302],
    [ 2.96102383, 2.2598307,  -0.69940133],
    [ 2.74560278, 3.76296751, -0.88668057],
    [ 3.47980754, 4.53616952, -0.10008854],
    [ 3.3316362,  5.98996057, -0.12287197],
    [ 4.58355452, 6.60851816, -0.74816152],
    [ 4.35459731, 7.60463118, -1.59134741],
    [ 5.41907018,  8.53327984, -1.96616928],
    [ 5.25395647,  9.82842507, -1.16852507],
    [ 4.60531723, 10.79606033, -1.79985236],
    [ 4.1430318,  11.98367674, -1.08442124],
    [ 2.85997035, 11.63986009, -0.32518253],
    [ 1.97381949, 10.93510234, -1.01342535],
    [ 0.74171301, 10.45861241, -0.38824281],
    [ 0.49119583,  9.01228285, -0.81983514],
    [-0.39369909,  8.35404832, -0.08545561],
    [-0.60984724,  6.92076923, -0.27246707],
    [-1.16458277,  6.67987328, -1.67786297],
    [-0.73789141,  5.57186797, -2.26598176],
    [-1.3578372 ,  5.08654901, -3.49728797],
    [-1.37242548,  3.5567375 , -3.47821136],
    [-2.26650212,  2.9952417 , -4.27882014],
    [-2.33374215,  1.5432591 , -4.43116038],
    [-1.5556844 ,  1.13733803, -5.68445773],
    [-0.36472043,  0.60353267, -5.45580591],
    [ 0.55736657,  0.32189981, -6.55417777],
    [ 0.23405732, -1.05612331, -7.13505936],
    [-0.49009014, -1.83929687, -6.34900934],
    [-0.7463319 , -3.23337059, -6.70521277],
    [-2.09860022, -3.65483349, -6.12673251]
  ]

  # Convert to our sin/cos phi/psi representation
  target_angles = []
  for angle in target_base:
    target_angles.append(math.sin(math.radians(angle[0])))
    target_angles.append(math.cos(math.radians(angle[0])))
    target_angles.append(math.sin(math.radians(angle[1])))
    target_angles.append(math.cos(math.radians(angle[1])))
    #target_angles.append(math.sin(math.radians(angle[2])))
    #target_angles.append(math.cos(math.radians(angle[2])))

  # pad out to max_cdr_length
  for i in range(len(target_angles), 28 * 4):
    target_angles.append(0.0)

  # We simulate a batch by doubling up
  target_angles = [target_angles, target_angles]
  target_angles = np.array( target_angles, dtype=np.float32)
  # Lengths are doubled as well
  lengths = [11,11] 
  ik = InverseK(28,2,4)
  (targets, results) = ik.train(target_angles, lengths, record=True)

  name = "3NH7_1"
  residues = []

  residues.append(repping.Residue(acids.AminoShort.GLU, 0, "0", math.radians(target_base[0][0]), math.radians(target_base[0][1]), math.radians(173.209)))
  residues.append(repping.Residue(acids.AminoShort.ARG, 1, "1", math.radians(target_base[1][0]), math.radians(target_base[1][1]), math.radians(-176.98)))
  residues.append(repping.Residue(acids.AminoShort.TRP, 2, "2", math.radians(target_base[2][0]), math.radians(target_base[2][1]), math.radians(162.29)))
  residues.append(repping.Residue(acids.AminoShort.HIS, 3, "3", math.radians(target_base[3][0]), math.radians(target_base[3][1]), math.radians(167.885)))
  residues.append(repping.Residue(acids.AminoShort.VAL, 4, "4", math.radians(target_base[4][0]), math.radians(target_base[4][1]), math.radians(175.878)))
  residues.append(repping.Residue(acids.AminoShort.ARG, 5, "5", math.radians(target_base[5][0]), math.radians(target_base[5][1]), math.radians(-172.224)))
  residues.append(repping.Residue(acids.AminoShort.GLY, 6, "6", math.radians(target_base[6][0]), math.radians(target_base[6][1]), math.radians(168.896)))
  residues.append(repping.Residue(acids.AminoShort.TYR, 7, "7", math.radians(target_base[7][0]), math.radians(target_base[7][1]), math.radians(175.872)))
  residues.append(repping.Residue(acids.AminoShort.PHE, 8, "8", math.radians(target_base[8][0]), math.radians(target_base[8][1]),math.radians(172.34)))
  residues.append(repping.Residue(acids.AminoShort.ASP, 9, "9", math.radians(target_base[9][0]), math.radians(target_base[9][1]), math.radians(-172.979)))
  residues.append(repping.Residue(acids.AminoShort.HIS, 10, "10", math.radians(target_base[10][0]), math.radians(target_base[10][1]), 0))

  print("Difference between computed and actual.")
  for i in range(0,len(actual_positions)):
    print(targets[0][i], actual_positions[i], "dif:", np.subtract(targets[0][i],actual_positions[i])) 

  from copy import deepcopy
  num_frames = len(results)
  frames = []

  for i in range(0, num_frames):
    frame = deepcopy(residues)
    for res in range(0, len(frame)):
      residue = frame[res]
      if res != 0:
        residue._phi = results[i][0][res * 2 - 1]
      else:
        residue._phi = 0
      if res < len(frame) - 1:
        residue._omega = math.pi #angle_range(math.degrees(float(angles[res * 3 + 1])))
        residue._psi = results[i][0][res * 2]
      else:
        residue._omega = 0
        residue._psi = 0

    frames.append(frame)
 
  from anim_json import *
  to_json_animation(name, frames)
  to_json_target(name, residues)
