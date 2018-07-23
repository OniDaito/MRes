def gen_worst(rmsds, pairs):
  ''' Find out what is making the worst offenders the worst.'''
  worst = {}
  final_rmsds = {}
  worst_acids = {}
  num_worst = {}
  for acid in amino_acid_bitmask:
    worst_acids[acid] = 0

  for rmsd in rmsds:
    key = os.path.basename(rmsd[0])
    key = key.split("_")[0]
    worst[key] = []
    final_rmsds[key] = rmsd[3]
    num_worst[key] = 0

  for pair in pairs:
    residues = pair_to_diff(pair)
    key = os.path.basename(pair[0]).split("_")[0]
    idx = 0
    res_errors = []
    for res in residues:
    
      phi0 = res[1] 
      phi1 = res[3]
      psi0 = res[2]
      psi1 = res[4]

      phi0 = (math.degrees(phi0) + 180) % 360
      phi1 = (math.degrees(phi1) + 180) % 360
      psi0 = (math.degrees(psi0) + 180) % 360
      psi1 = (math.degrees(psi1) + 180) % 360

      diff_phi = min(phi1-phi0, 360 + phi0 - phi1)
      if phi1 < phi0:
        diff_phi = min(phi0-phi1, 360 + phi1 - phi0)
      
      diff_psi = min(psi1-psi0, 360 + psi0 - psi1)
      if psi1 < psi0:
        diff_psi = min(psi0-psi1, 360 + psi1 - psi0)

      res_errors.append((idx,diff_phi,diff_psi,res[0]))

      idx += 1
    
    worst[key] = res_errors

  worst_keys = []
  for key in worst.keys():
    try:
      if final_rmsds[key] > 2.0:
        worst_keys.append(key)
    except:
      pass

  ANGLE = 120
  for key in worst_keys:
    for res in worst[key]:
      if res[0] >= ANGLE or res[1] >= ANGLE:
        worst_acids[res[3]] += 1
        num_worst[key] += 1

  print("Worst Keys and Percentages")
  for key in worst_keys: 
    print(key, final_rmsds[key], num_worst[key], len(worst[key]), float(num_worst[key]) / float(len(worst[key])) * 100.0)

  print("Worst as percentage:", len(worst_keys) / len(pairs) * 100.0)

  print("Worst Acids")
  print(worst_acids)
  return (worst_keys, worst,final_rmsds)


