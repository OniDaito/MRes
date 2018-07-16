import math

def get_angles(line):
  t0 = line.split(":")
  t1 = t0[1].split(",")
  residue = t0[0]
  phi = float(t1[0])
  psi = float(t1[1])
  return (residue,phi,psi)

def do_compare(file0, file1):
  f0 = []
  with open(file0,'r') as f:
    f0 = f.readlines()

  f1 = []
  with open(file1,'r') as f:
    f1 = f.readlines()
  
  tgraph = []

  for i in range(0,len(f0)):
    res, phi0, psi0 = get_angles(f0[i]) 
    _, phi1, psi1 = get_angles(f1[i])
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

    tgraph.append((res,max(diff_phi,diff_psi)))

    #print("***" + res + "***")
    #print(phi0, psi0, phi1, psi1)
    #print(diff_phi, diff_psi)

  from ascii_graph import Pyasciigraph
  graph = Pyasciigraph(
        line_length=80,
        min_graph_length=0,
        force_max_value=180,
      )
  for line in  graph.graph('max diff per residue', tgraph):
    print(line) 



if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Compare two files')
  parser.add_argument('file0', metavar='A', type=str, help='path to neural net directory')
  parser.add_argument('file1', metavar='B', type=str, help='path to neural net directory')

  args = vars(parser.parse_args())
  if args['file0'] and args['file1']:
    do_compare(args['file0'], args['file1'])

