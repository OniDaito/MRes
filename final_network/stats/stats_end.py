def _gen_end(path):
  parser = PDBParser()
  distances = []
  bn = os.path.basename(path)
  try:
    st = parser.get_structure(bn, path)  
  except:
    print("Failed to load",path)
    return -1

  models = st.get_models() # should only be one
  num_residues = 0
  for model in models:
    atoms = []
    for atom in model.get_atoms():
      atoms.append(atom)

    # Should just be CAs
    x0 = atoms[0].get_coord()[0]
    y0 = atoms[0].get_coord()[1]
    z0 = atoms[0].get_coord()[2]
        
    x1 = atoms[-1].get_coord()[0]
    y1 = atoms[-1].get_coord()[1]
    z1 = atoms[-1].get_coord()[2]

    dist = math.sqrt( (x0-x1) * (x0-x1) + (y0-y1) * (y0-y1) + (z0-z1) * (z0-z1))
    distances.append(dist)
  return distances

def gen_end(pairs):
  dp = []
  dr = []
  for pair in pairs:
    # 0 is predicted, 1 is real
    d_pred = _gen_end(pair[0])
    d_real = _gen_end(pair[1])
    if d_pred == -1 or d_real == -1:
      continue
    dp.append(d_pred[0])
    dr.append(d_real[0]) # should only be one

  distances = list(zip(dp,dr))

  # Lets return some stats here
  mn = 7.537065927577005
  sd = 1.13033939846 * 2.0
  
  valid_r = 0
  valid_p = 0
  both_valid = 0
  both_wrong = 0
  real_right = 0
  pred_right = 0
  
  new_pairs = []

  idx = 0
  for dd in distances:
    vp = False
    vr = False
    if not(dd[0] > mn + sd or dd[0] < mn - sd):
      valid_p += 1
      new_pairs.append(pairs[idx])
      vp = True

    if not(dd[1] > mn + sd or dd[1] < mn - sd):
      valid_r += 1
      vr = True
    #else:
    #  print(pairs[idx])

    if vr and vp:
      both_valid += 1
    elif not vr and not vp:
      both_wrong += 1
    elif vr and not vp:
      real_right += 1
    elif not vr and vp:
      pred_right += 1
    idx += 1

  total = len(distances)

  # Bit nasty!
  return (valid_r, valid_p, total, new_pairs)


