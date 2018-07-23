"""
batcher.py - converting and batching our data for use in the nets
author : Benjamin Blundell
email : me@benjamin.computer

Take the data from the database and create a set of batches in both
loop and numpy form to feed to our nets for training.

"""

import os, math, random
from random import randint 
import numpy as np
from enum import Enum

if __name__ != "__main__":
  parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.sys.path.insert(0,parentdir)
  import common.acids as acids

SetType = Enum("SetType","TRAIN TEST VALIDATE")
BatchTypeIn = Enum("BatchTypeIn","BITFIELD BITFIELDTRIPLE FIVED FIVEDTRIPLE FIVEDADD SEQ")
BatchTypeOut = Enum("BatchTypeOut","SINCOS CAT SEQ")

# If we use the compact 5D notation, here are the lookups
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

def get_cat(vec):
  maxp = 0
  idm = 0 # 0 for no class / mask out I think. If using tf.sparse logits
  for i in range(0,len(vec)):
    if vec[i] > maxp:
      maxp = vec[i]
      idm = i + 1 
  return idm

def angles_to_cat(phi, psi, as_vec=True):
  ''' Return a category for our phi / psi combo. We limit to areas of the ramachandran
  plot, with the 0 category being an unknown our outlier. Categories overlap so we
  return a set of probabilities. We use 10 degree blocks giving 36 categories.'''

  phi_cat = int(math.floor((math.degrees(phi) + 180) / 10))
  psi_cat = int(math.floor((math.degrees(psi) + 180) / 10))
  cat = min(36*36-1, phi_cat * 36 + psi_cat)

  if as_vec:
    vec = []
    for i in range(0,36*36):
      vec.append(0)
    vec[cat] = 1.0

    return vec
  return cat

def cat_to_angles(cv, vec=True):
  ''' Do a sort of reduce max to find the most likely category.'''
  cat = cv
  if vec:
    maxc = 0
    cat = 0
    for i in range(0,len(vec)):
      if vec[i] > maxc:
        maxc = vec[i]
        cat = i

  phi = math.radians((int(math.floor( cat / 36)) * 10) - 180)
  psi = math.radians((int(math.floor( cat % 36)) * 10) - 180)
  return (phi,psi)


def trip_to_incat(trip):
  ''' Take three aminos (or two + empty) and return a numeric category. 
  We make things simple by allowing the blank anywhere. Total number of
  classes is 21 * 21 * 21 = 9261'''
  
  aa = ["BLK"] + acids.TRIPLES

  t0 = [i for i,x in enumerate(aa) if x == trip[0]][0] 
  t1 = [i for i,x in enumerate(aa) if x == trip[1]][0] 
  t2 = [i for i,x in enumerate(aa) if x == trip[2]][0]

  # We add a +1 as we use 0 as the no-class padding
  cls =  int((math.pow(21,2) * t0) + (21 * t1) + (t2)) + 1 
  return cls

def tripcat_to_acid(cat):
  ''' Convert from incat back to an acid.'''
  aa = ["BLK"] + acids.TRIPLES
  cat = cat - 1
  t0 = int(math.floor(cat / math.pow(21,2)))
  cls = cat - (t0 * math.pow(21,2))
  t1 = int(math.floor(cls / 21))
  cls = cls - (t1 * 21)
  t2 = int(math.floor(cls))
  if t1 == 0:
    return None
  return acids.label_to_amino(aa[t1])

def normalise(va):
  l = math.sqrt(sum([a * a for a in va]))
  return [ a/l for a in va]

class Batcher(object): 
  ''' Create numpy batches of data, with a bitmask and phi/psi angles. 
  We default to three datasets: train, test and validate.'''
  def __init__(self, batch_size = 20, max_cdr_length=28, typein=BatchTypeIn.BITFIELD, typeout=BatchTypeOut.SINCOS, datasize=10):
    # Will be filled with instances of Loop
    self.train_set = []
    self.validate_set = []
    self.test_set = []
    self._output_width = 4
    self.datasize = datasize
    self._avail_train = [] # record the datapoints still available (for random batches)
    self._avail_validate = []
    self._avail_test = []
    self.type_in = typein
    self.type_out = typeout

    # Will be a tuple of two numpy arrays, input and output. Our Labels basically.
    self.train_set_np = None
    self.validate_set_np = None
    self.test_set_np = None
    
    self.batch_size = batch_size
    self._next_batch = 0
    self.max_cdr_length = max_cdr_length

  def _acid_bit(self, res):
    ta = []
    for n in range(0, acids.NUM_ACIDS):
      ta.append(False)
     
    mask = acids.amino_pos(res._name)
    ta[mask] = True
    return ta

  def input_to_acid(self, ip):
    if self.type_in == BatchTypeIn.FIVED:
      return acids.vector_to_acid(ip)
    elif self.type_in == BatchTypeIn.SEQ:
      return tripcat_to_acid(ip)
    else:
      return acids.bitmask_to_acid(ip) 
    return None

  def create_mask(self, batch):
    ''' create a mask for our fully connected, final output layer
    for either sin/cos phi/psi representation or classification.'''
    mask = []
   
    for model in batch:
      mm = []
      for res in model:
   
        if self.type_out == BatchTypeOut.SINCOS:
          mt = []
          if sum(res) == 0:
            for i in range(0,4):
              mt.append(0)
          else:
            for i in range(0,4):
              mt.append(1)
      
        elif self.type_out == BatchTypeOut.CAT:
          mt = [] 
          if sum(res) == 0: # Should work for bitfield and 5D
            for i in range(0,36*36):
              mt.append(0)
          else:
            for i in range(0,36*36):
              mt.append(1)
    
        elif self.type_out == BatchTypeOut.SEQ:
          mt = 0
          if res != 0:
            mt  = 1
  
        mm.append(mt)

      if self.type_out == BatchTypeOut.SEQ:
        mm = [1] + mm
      mask.append(mm)
    
    return np.array(mask,dtype=np.float32)

  def _to_numpy_input(self, loops):
    ''' Convert our loops into the numpy input vectors. Depends on the
    self.batchtype variable.'''
    
    ft = []
    for loop in loops:
      tt = []

      if self.type_in == BatchTypeIn.BITFIELD:
        # Standard bitfield of acids size * residues
        for i in range(0, self.max_cdr_length):
          tt.append([False for x in range(0, acids.NUM_ACIDS)])
        
        idx = 0
        for residue in loop.get_residues():
          ta = self._acid_bit(residue)
          tt[idx] = ta
          idx += 1
        ft.append(tt)

      elif self.type_in == BatchTypeIn.BITFIELDTRIPLE:
        for i in range(0, self.max_cdr_length):
          tt.append([False for x in range(0, acids.NUM_ACIDS * 3)])
        
        for idr in range(0,len(loop.get_residues())):
          # Centre residue
          residue = loop.get_residues()[idr] 
          ta = self._acid_bit(residue)
          for i in range(0, acids.NUM_ACIDS):
            tt[idr][i+acids.NUM_ACIDS] = ta[i]
        
          if idr > 0:
            # lef residue
            residue = loop.get_residues()[idr-1] 
            ta = self._acid_bit(residue)
            for i in range(0, acids.NUM_ACIDS):
              tt[idr][i] = ta[i]
          
          if idr < len(loop.get_residues()) - 1:
            # right residue
            residue = loop.get_residues()[idr+1] 
            ta = self._acid_bit(residue)
            for i in range(0, acids.NUM_ACIDS):
              tt[idr][i + (acids.NUM_ACIDS * 2)] = ta[i]

        ft.append(tt)

      elif self.type_in == BatchTypeIn.FIVED:
        # Five input numbers per residue per loop
        for i in range(0, self.max_cdr_length):
          tt.append([0,0,0,0,0])
        idx = 0
        
        for residue in loop.get_residues():
          tt[idx] = normalise(lookup_table[acids.amino_to_label(residue._name)])
          idx+=1 

        ft.append(tt)

      elif self.type_in == BatchTypeIn.SEQ:
        # Five input numbers per residue per loop
        for i in range(0, self.max_cdr_length):
          tt.append(0) # 0 is the end stop
                
        reses = loop.get_residues()
        for idx in range(0,len(reses)):
          trip = ["BLK",acids.amino_to_label(reses[idx]._name),"BLK"]
          if idx > 0:
            trip[0] = acids.amino_to_label(reses[idx-1]._name)
          if idx < len(reses)-1:
            trip[2] = acids.amino_to_label(reses[idx+1]._name)
          
          tt[idx] = trip_to_incat(trip)

        ft.append(tt)

      elif self.type_in == BatchTypeIn.FIVEDTRIPLE:
        # 15 input numbers per residue per loop
        for i in range(0, self.max_cdr_length):
          tt.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
        for idr in range(0,len(loop.get_residues())):
          # Centre residue
          residue = loop.get_residues()[idr]
          tv = lookup_table[acids.amino_to_label(residue._name)]
          
          if idr > 0:
            residue = loop.get_residues()[idr-1]
            tv2 = lookup_table[acids.amino_to_label(residue._name)]
            for i in range(0,5):
              tt[idr][i] = tv2[i]
     
          if idr < len(loop.get_residues())-1:
            residue = loop.get_residues()[idr+1]
            tv2 = lookup_table[acids.amino_to_label(residue._name)]
            for i in range(0,5):
              tt[idr][10+i] = tv2[i]
        
          for i in range(0,5):
            tt[idr][5+i] = tv[i]
      
        ft.append(tt)
      
      elif self.type_in == BatchTypeIn.FIVEDADD:
        # Look at the triples but add them together
        for i in range(0, self.max_cdr_length):
          tt.append([0,0,0,0,0])
      
        for idr in range(0,len(loop.get_residues())):
          # Centre residue
          residue = loop.get_residues()[idr]
          tv = lookup_table[acids.amino_to_label(residue._name)]
          
          if idr > 0:
            residue = loop.get_residues()[idr-1]
            tv2 = lookup_table[acids.amino_to_label(residue._name)]
            for i in range(0,5):
              tt[idr][i] += tv2[i]
     
          if idr < len(loop.get_residues())-1:
            residue = loop.get_residues()[idr+1]
            tv2 = lookup_table[acids.amino_to_label(residue._name)]
            for i in range(0,5):
              tt[idr][i] += tv2[i]
        
          for i in range(0,5):
            tt[idr][i] += tv[i]
       
        ft.append(tt)
    
    input_set = np.array(ft,dtype=np.float32)
    if self.type_in == BatchTypeIn.SEQ:
      input_set = np.array(ft,dtype=np.int32)
    return input_set
  
  def _to_numpy_output(self, loops):
    ''' Convert our loops into the numpy output vectors.'''
    fu = []
  
    for loop in loops:
      tu = []
    
      if self.type_out == BatchTypeOut.SINCOS:

        for i in range(0, self.max_cdr_length):
          tt = []
          for j in range(0,4):
            tt.append(-3.0)
          tu.append(tt)

        idx = 0
        for residue in loop.get_residues():
          tu[idx][0] = math.sin(residue._phi)
          tu[idx][1] = math.cos(residue._phi)
          tu[idx][2] = math.sin(residue._psi)
          tu[idx][3] = math.cos(residue._psi)
          idx += 1
      
      elif self.type_out == BatchTypeOut.CAT:    
        for i in range(0, self.max_cdr_length):
          tt = []
          for j in range(0, 36 * 36):
            tt.append(0)
          tu.append(tt)

        idx = 0
        for residue in loop.get_residues():
          tu[idx] = angles_to_cat(residue._phi, residue._psi)
          idx += 1
      
      elif self.type_out == BatchTypeOut.SEQ:    
        for i in range(0, self.max_cdr_length+1):
          tt = 0
          tu.append(tt)
        tu[0] = 36 * 36 + 1 # Our <s> equiv
        
        idx = 1
        for residue in loop.get_residues():
          tu[idx] = angles_to_cat(residue._phi, residue._psi, as_vec=False)
          idx += 1

      fu.append(tu)

    #print("Output Vector Test:", len(fu), len(fu[0]))
    output_set = np.array(fu,dtype=np.float32)
    if self.type_out == BatchTypeOut.SEQ:
      output_set = np.array(fu,dtype=np.int32)
    return output_set


  def partial_pickle_it(self, datasets, filename):
    '''Just a little bit.'''
    import pickle
    with open(filename, 'wb') as f:
      print("Partial Pickling data", filename)
      pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)


  def create_sets_from_loops(self, loops, filename="data.pickle", blockers=[], ignores=[], train_size=80, validate_size=10, test_size=10, partial_pickle=False):
    ''' Grab data with grabber, then partition given the sizes (as percentage).
    Blockers are redundant items that should not appear in the validation or test sets.
    Ignores should not appear in any set.'''
    # TODO - loops could be quite large. Might need to do multiple batches with offsets in grabber somehow?
    
    import gc
    total = len(loops)
 
    tloops = []
    aloops = []
    for i in range(0,len(loops)):
      tloops.append(i)

    # Build validate and test first so we prioritise non-blockers
    # Build validate set
    validate_len = min( int(math.floor(total * 0.66)), int(math.floor(validate_size / 100 * total)))
    
    while len(self.validate_set) < validate_len and len(tloops) > 0:
      tidx = randint(0,len(tloops)-1) 
      lidx = tloops[tidx]
      if loops[lidx]._name not in ignores and loops[lidx]._name not in blockers:
        self.validate_set.append(loops[lidx])
        aloops.append(lidx)
      del tloops[tidx]

    for lidx in sorted(aloops, reverse=True):
      del loops[lidx]
 
    self.validate_set_np = (
      self._to_numpy_input(self.validate_set),
      self._to_numpy_output(self.validate_set)
    )
    for i in range(0,len(self.validate_set)):
      self._avail_validate.append(i)

    print("Created validation set of size: ", len(self.validate_set))
    
    if partial_pickle:
      self.partial_pickle_it([self.validate_set_np[0], self.validate_set_np[1], self.validate_set], filename + "_valid")
      del self.validate_set
      del self.validate_set_np
      gc.collect()

    # Build Test set
    tloops = []
    aloops = []
    for i in range(0,len(loops)):
      tloops.append(i)

    test_len = min( int(math.floor(total * 0.33)), int(math.floor(test_size / 100 * total)))
    while len(self.test_set) < test_len and len(tloops) > 0:
      tidx = randint(0,len(tloops)-1)
      lidx = tloops[tidx]
      if loops[lidx]._name not in ignores and loops[lidx]._name not in blockers:
        self.test_set.append(loops[lidx])
        aloops.append(lidx)
      del tloops[tidx]

    for lidx in sorted(aloops, reverse=True):
      del loops[lidx]
 
    gc.collect()

    self.test_set_np = (
        self._to_numpy_input(self.test_set),
        self._to_numpy_output(self.test_set)
      )
    
     # set the avail lists
    for i in range(0,len(self.test_set)):
      self._avail_test.append(i)
   
    print("Created test set of size: ", len(self.test_set))
    if partial_pickle:
      self.partial_pickle_it([self.test_set_np[0], self.test_set_np[1], self.test_set], filename + "_test")
      del self.test_set
      del self.test_set_np
      gc.collect()

    # Build train set - for now lets go with everything that is left
    #train_size = int(math.floor(train_size / tt * total))
    train_size = len(loops)

    # Partial divides into 10 pieces as it's too big otherwise
    if partial_pickle:
      width = int(math.floor(train_size / self.datasize))
      extra = int(train_size % self.datasize)

      for j in range(0,self.datasize):
        train_set = []

        for i in range(j*width, (j+1)*width):
          lidx = randint(0,len(loops)-1)
          if loops[lidx]._name not in ignores:
            train_set.append(loops[lidx])
          del loops[lidx]
     
        # We have to do each set individually
        ti = self._to_numpy_input(train_set),
        self.partial_pickle_it(ti, filename + "_train_in_" + str(j))
        del ti
        gc.collect()

        to = self._to_numpy_output(train_set)
        self.partial_pickle_it(to, filename + "_train_out_" + str(j))
        del to
        gc.collect()

        self.partial_pickle_it(train_set, filename + "_train_" + str(j))
        del train_set
        gc.collect()

    else:
      for i in range(0,train_size):
        lidx = randint(0,len(loops)-1)
        if loops[lidx]._name not in ignores:
          self.train_set.append(loops[lidx])
        del loops[lidx]
     
      for i in range(0,len(self.train_set)):
        self._avail_train.append(i)

      self.train_set_np = ( 
          self._to_numpy_input(self.train_set),
          self._to_numpy_output(self.train_set)
        )
    
    print("Created training set of size: ", len(self.train_set))
     
  def create_sets_from_pickle(self, filename, partial_pickle=True):
    ''' Load the data from a pickle file. '''
    import pickle
    print("Loading from pickle file")
    
    train = None
    val = None
    test = None
    train_in = None
    train_on = None
    val_in = None
    val_on = None
    test_in = None
    test_on = None

    if partial_pickle:
      train = None
      with open(filename + "_train_0", 'rb') as f:
        train = pickle.load(f)
      for i in range(1,self.datasize):
        with open(filename + "_train_" + str(i), 'rb') as f:
          train += pickle.load(f)

      train_in = None
      with open(filename + "_train_in_0", 'rb') as f:
        train_in = pickle.load(f)[0]
      
      for i in range(1,self.datasize):
        with open(filename + "_train_in_" + str(i), 'rb') as f:
          train_in = np.concatenate((train_in, pickle.load(f)[0]), axis=0)
     
      train_on = None
      with open(filename + "_train_out_0", 'rb') as f:
        train_on = pickle.load(f)

      for i in range(1,self.datasize):
        with open(filename + "_train_out_" + str(i), 'rb') as f:
          train_on = np.concatenate((train_on, pickle.load(f)), axis=0)
  
      with open(filename + "_valid", 'rb') as f:
        val_in, val_on, val = pickle.load(f)
      
      with open(filename + "_test", 'rb') as f:
        test_in, test_on, test = pickle.load(f)
     
    else:    
      with open(filename, 'rb') as f:
        train, val, test, train_in, train_on, val_in, val_on, test_in, test_on = pickle.load(f)
    
    print("Loaded training loop/in/out sets of size", len(train), train_in.shape, train_on.shape)

    self.train_set = train
    self.validate_set = val
    self.test_set = test

    self.train_set_np = (train_in, train_on)
    self.test_set_np = (test_in, test_on)
    self.validate_set_np = (val_in, val_on)

    self.max_cdr_length = len(train_in[0])
    self._output_width = int(math.floor(train_on.shape[1] / self.max_cdr_length))

  def has_next_batch(self, settype):
    ''' Is there enough left for another batch?''' 
    input_data = self.train_set_np[0]
    if settype == SetType.TEST: input_data = self.test_set_np[0]
    elif settype == SetType.VALIDATE: input_data = self.validate_set_np[0]

    s = self._next_batch
    v = self.batch_size
    if s + v >= len(input_data):
      return False
    return True

  def batch_to_labels(self, outputb):
    labels = []
    for b in outputb:
      cdr = []
      for r in b:
        cdr.append(get_cat(r))
      labels.append(cdr)
    return np.array(labels,dtype=np.int32)

  def reset(self):
    ''' Start again from the top.'''
    random.seed()
    self._next_batch = 0
    del self._avail_test[:]
    del self._avail_train[:]
    del self._avail_validate[:]
    # set the avail lists
    for i in range(0,len(self.test_set)):
      self._avail_test.append(i)
    for i in range(0,len(self.train_set)):
      self._avail_train.append(i)
    for i in range(0,len(self.validate_set)):
      self._avail_validate.append(i)

  def get_set(self, settype):
    if settype == SetType.TEST:
      return self.test_set_np
    elif settype == SetType.VALIDATE:
      return self.validate_set_np
    return self.train_set_np

  def get_loop_set(self, settype):
    if settype == SetType.TEST:
      return self.test_set
    elif settype == SetType.VALIDATE:
      return self.validate_set
    return self.train_set

  def get_avail(self, settype):
    if settype == SetType.TEST:
      return self._avail_test
    elif settype == SetType.VALIDATE:
      return self._avail_validate
    return self._avail_train

  def offset(self, off, settype):
    (input_data, output_data) = self.get_set(settype)
    if off < len(input_data):
      self._next_batch = off
    else:
      print("Offset greater than dataset size")

  def has_next_batch_random(self, settype):
    avail = self.get_avail(settype)
    return len(avail) >= self.batch_size

  def next_batch(self, settype, randset=False, pad=True): 
    s = self._next_batch
    v = self.batch_size
    (input_data, output_data) = self.get_set(settype)
    loop_data = self.get_loop_set(settype)

    ix = []
    ox = []  
    choices = []
    max_len = 0

    if randset:
      avail = self.get_avail(settype)

      for b in range(v):
        r = randint(0,len(avail)-1)
        choices.append(r)
        l = len(loop_data[r].get_residues())
        if l > max_len:
          max_len = l
        del avail[r]

    pad_len = self.max_cdr_length
    if not pad:
      pad_len = max_len
    
    ix = np.zeros((v, pad_len, acids.NUM_ACIDS))
    ox = np.zeros((v, pad_len, 4))
  
    if self.type_in == BatchTypeIn.FIVED or self.type_in == BatchTypeIn.FIVEDADD:
      ix = np.zeros((v, pad_len, 5))

    elif self.type_in == BatchTypeIn.SEQ:
      ix = np.zeros((v, pad_len))

    elif self.type_in == BatchTypeIn.FIVEDTRIPLE:
      ix = np.zeros((v, pad_len, 15))
    
    elif self.type_in == BatchTypeIn.BITFIELDTRIPLE:
      ix = np.zeros((v, pad_len, acids.NUM_ACIDS * 3))

    if self.type_out == BatchTypeOut.CAT:
      ox = np.zeros((v, pad_len, 36 * 36))
    
    elif self.type_out == BatchTypeOut.SEQ:
      pad_len += 1
      ox = np.zeros((v, pad_len))

    loops = []
  
    if randset:
      for b in range(v):
        r = choices[b]
        ix[b] = input_data[r][:pad_len]
        ox[b] = output_data[r][:pad_len]
        loops.append (loop_data[r])
    
    else:
      for b in range(v):
        ix[b] = input_data[s + b][:pad_len]
        ox[b] = output_data[s + b][:pad_len]
        loops.append (loop_data[s + b])
    
      
    self._next_batch = s + v
    return (ix, ox, loops)

  def max_pad_out(self, batch_out):
    ''' This creates a set of labels for SEQ. We pad to the maximum length
    then shift left one, adding a 0 at the end as out </s>'''
    # For now just work with seq out
    assert self.type_out == BatchTypeOut.SEQ
    max_pad = 0

    for b in batch_out:
      bl = 0
      for i in b:
        if i == 0:
          bl += 1
    
      bl = self.max_cdr_length + 1 - bl
      
      if bl > max_pad:
        max_pad = bl

    v = self.batch_size
    #ox = np.zeros((v, max_pad))
    ox = []
    for i in range(0,v):
      oy = []
      for j in range(1,max_pad):
        oy.append(batch_out[i][j])
      oy.append(0)
      ox.append(oy)

    ox = np.array(ox, dtype=np.int32)

    return ox

  def random_batch(self, settype, pad = True) : 
    ''' A random set of length batch_size from the input data.''' 
    ix = []
    ox = [] 
    v = self.batch_size

    (input_data, output_data) = self.get_set(settype)
    loop_data = self.get_loop_set(settype)
    loops = []
    choices = []
    max_len = 0


    for i in range(0,self.batch_size):
      rnd = randint(0,len(input_data)-1) 
      choices.append(rnd)
      l = len(loop_data[rnd].get_residues())
      if l > max_len:
        max_len = l
     
    pad_len = self.max_cdr_length
    if not pad:
      pad_len = max_len

    ix = np.zeros((v, pad_len, acids.NUM_ACIDS))
    ox = np.zeros((v, pad_len, 4))
    
    if self.type_in == BatchTypeIn.FIVED or self.type_in == BatchTypeIn.FIVEDADD:
      ix = np.zeros((v, pad_len, 5))
   
    elif self.type_in == BatchTypeIn.SEQ:
      ix = np.zeros((v, pad_len))

    elif self.type_in == BatchTypeIn.FIVEDTRIPLE:
      ix = np.zeros((v, pad_len, 15))
    
    elif self.type_in == BatchTypeIn.BITFIELDTRIPLE:
      ix = np.zeros((v, pad_len, acids.NUM_ACIDS * 3))

    if self.type_out == BatchTypeOut.CAT:
      ox = np.zeros((v, pad_len, 36 * 36))

    if self.type_out == BatchTypeOut.SEQ:
      pad_len += 1
      ox = np.zeros((v, pad_len))

    b = 0
    for rnd in choices:
      ix[b] = input_data[rnd][:pad_len]
      ox[b] = output_data[rnd][:pad_len]
      loops.append(loop_data[rnd])
      b+=1

    return (ix, ox, loops)

  def pickle_it(self, filename):
    '''Just a little bit.'''
    import pickle
    with open(filename, 'wb') as f:
      print("Pickling data...")
      dataset = [self.train_set, self.validate_set, self.test_set, 
          self.train_set_np[0], self.train_set_np[1],
          self.validate_set_np[0], self.validate_set_np[1],
          self.test_set_np[0], self.test_set_np[1]]

      pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  import acids, gen_data 
  import argparse
  parser = argparse.ArgumentParser(description='Create Loop batches')
  parser.add_argument('--both', action='store_true', help='generate pdbs from test set')
  parser.add_argument('--nobad', action='store_true', help='Remove bad loops')
  parser.add_argument('--deets', action='store_true', help='Read sets and print stats')
  parser.add_argument('--size', type=int, help='Total set size from loopdb')
  parser.add_argument('--max', type=int, help='Max CDR length')
  parser.add_argument('--min', type=int, help='Min CDR length')
  parser.add_argument('--block', action='store_true', help='Block redundants')
  parser.add_argument('--fixed', type=int, help='Only one size of CDR')
  parser.add_argument('--typein', type=int, help='Which type of batch? [0 bitfield, 1 bittrip, 2 fived, 3 fivedtrip, 4 fiveadd]')
  parser.add_argument('--typeout', type=int, help='Which type of batch? [0 sin/cos/phi/psi 1 cat]')
  parser.add_argument('--out', type=str, help='Name of the output file')
  

  args = parser.parse_args()
  #g = gen_data.Grabber(remove_bad_endpoint = args.nobad, remove_odd_omega = args.nobad)
  # AMA ignore list
  ama_list = ["4MA3_1", "4MA3_2","4KUZ_1", "4KQ3_1", "4KQ4_1", "4M6M_1", "4M6O_1", "4MAU_1", "4M7K_1", "4KMT_1", "4M61_1", "4M61_2", "4M43_1"]

  mincdr = 0
  maxcdr = 32
  if args.fixed:
    mincdr = int(args.fixed)
    maxcdr = mincdr
  else:
    if args.max:
      maxcdr = int(args.max)
    if args.min:
      mincdr = int(args.min)

  typein = BatchTypeIn.BITFIELD
  if args.typein:
    typein = list(BatchTypeIn)[int(args.typein)]

  typeout = BatchTypeOut.SINCOS
  if args.typeout:
    typeout = list(BatchTypeOut)[int(args.typeout)]

  out_name = "data.pickle"
  if args.out:
    out_name = str(args.out)

  # Details check - print sizes
  if args.deets and args.max and args.out:
    print("Reading details of pickles.")
    d = Batcher(max_cdr_length = maxcdr, typein = typein, typeout=typeout, batch_size = 2)
    d.create_sets_from_pickle(filename=out_name, partial_pickle = args.both)
    (ix,ox,loops) = d.random_batch(SetType.VALIDATE)
    print("Sizes (train, validate, test)", len(d.train_set), len(d.validate_set), len(d.test_set))
    import sys
    sys.exit()

  g = gen_data.Grabber(remove_bad_endpoint = args.nobad, remove_odd_omega = args.nobad, mongo=False)
  (loops, summary, blockers) = g.grab(min_length=mincdr, max_length=maxcdr, block=args.block)

  if args.both:
    g = gen_data.Grabber(remove_bad_endpoint = args.nobad, remove_odd_omega = args.nobad, mongo=True)
    ll = -1
    if args.size:
      ll = int(args.size)
    # No blockers from mongo as we've already filtered them out
    (loops_m, summary_m) = g.grab(limit=ll,min_length=mincdr, max_length=maxcdr, block=args.block)
    loops = loops + loops_m
  

  b = Batcher(max_cdr_length = maxcdr, typein = typein, typeout=typeout, datasize=50)
  b.create_sets_from_loops(loops, filename=out_name, blockers=blockers, ignores=ama_list, partial_pickle = args.both)
  if not args.both:
    b.pickle_it(out_name)

  # Test re-reading
  c = Batcher(max_cdr_length = maxcdr, typein = typein, typeout=typeout, batch_size = 2)
  c.create_sets_from_pickle(filename=out_name, partial_pickle = args.both)

  (ix,ox,loops) = c.random_batch(SetType.VALIDATE)
  print (ix)
  print (ox)
  print (c.create_mask(ix))
  print ("Loaded train/in/out size", len(c.train_set), c.train_set_np[0].shape, c.train_set_np[1].shape)


