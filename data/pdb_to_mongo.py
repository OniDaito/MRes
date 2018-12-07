"""
pdb_to_mongo.py - convert a series of PDB loops to data for mongo
author : Benjamin Blundell
email : me@benjamin.computer

Currently using biopython for this but I don't think we *really* need to.
Some PDBs actually don't contain torsion angles so we need to detect these
I think and reject.

"""

from Bio.PDB import *
import sys, time, pymongo, traceback, os, argparse

class PDBMongo ():

  def __init__(self):
    self.parser = PDBParser()
    self.conn = None

  def db_connect(self, dbname):
    """ Connect to the mongo database instance. """
    from pymongo import MongoClient
    self.conn = MongoClient()
    self.conndb = self.conn[dbname]

  def gen_angles(self):
    """ Generate the angles we will need for our neural net. """ 
    try:
      for model in self.conndb.models.find(\
          no_cursor_timeout=True).batch_size(3):
        mname = model["code"].replace(" ","")
        atoms = self.conndb.atoms.find({"model" : mname},\
            no_cursor_timeout=True)
   
        try: 
          # Extract the different residues 
          residues = []
          current_residue = [] # name, amino, num, x, y, z
          current_name = ""
          dd = 0
          previdx = atoms[0]['resseq']

          for atom in atoms:
            # model, num, label, space, amino, chain, res num, none, 
            # x, y, z, <rest not needed>
            if atom['resname'] != current_name or atom['name'] == 'N':
              current_name = atom['resname']
              if len(current_residue) != 0 :
                if int(atom['resseq']) == previdx:
                  dd += 1

                previdx = int(atom['resseq'])
                residues.append(current_residue.copy())
                current_residue = []
                dd = 0
        
            rr = (atom['name'], atom['resname'], int(atom['resseq']),\
                float(atom['x']), float(atom['y']), float(atom['z']))
            current_residue.append(rr)

          residues.append(current_residue)
          # We drop the first and last 3 as these are the anchor points
          # in pdb_loopdb apparently
          residues = residues[3:]
          residues = residues[:-3]
  
          if len(residues) < 3:
            continue

          # Write the residues we are considering into the db for speed
          idx = 0
          for res in residues:
            try:
              tres = { "model" : mname, "residue" : res[1][1],\
                  "reslabel" : res[1][2], "resorder" : idx }
              self.conndb.residues.insert_one(tres)
              
            except Exception as e:
              print("Error inserting into DB - " + model, e)
              traceback.print_exc()
              sys.exit()

            idx+=1
 
          # Now we have all the residues, work out the angles and write
          # to the database.
          angles = torsions.derive_angles(residues)
          idx = 0
          for aa in angles:
            try:
              tangle = {"model" : mname, "phi" : aa[0], \
                  "psi" : aa[1], "omega" : aa[2], "resorder" : idx }
              self.conndb.angles.insert_one(tangle)
            except Exception as e:
              print("Error inserting into DB - " + model, e)
            idx +=1
          
          sys.stdout.write("\n")
        except Exception as e:
          print(e)
          traceback.print_exc()
          sys.exit()

    except Exception as e:
        print ("Exception in DB read: ", e)
        traceback.print_exc()
        sys.exit()

  def _check(self, model):
    seq = ""
    for chain in model:
      chainID = chain.get_id()
      # Assuming there is only one chain! Bit naughty!
      for residue in chain:
        ## The test below checks if the amino acid
        ## is one of the 20 standard amino acids
        ## Some proteins have "UNK" or "XXX", or other symbols
        ## for missing or unknown residues
        if is_aa(residue.get_resname(), standard=True):
          seq += three_to_one(residue.get_resname())
        else:
          return (False, seq)

    atom_count = 0
    for atom in model.get_atoms():
      atom_count +=1
      if atom_count > 10:
        break
        
      if atom_count < 10:
        print("PDB Model contains no atoms.")
        return (False, seq)

    return (True, seq)
    
  def process_pdb(self, pdb_path):
    ''' Read a PDB and write to the DB.'''
    try:
      bn = os.path.basename(pdb_path),
      st = self.parser.get_structure(bn, pdb_path)
    except:
      print("Failed to parse:", pdb_path)
      return

    models = st.get_models()
    for model in models:
      mname = os.path.splitext(model.get_full_id()[0][0])[0]
      print("Working with ", mname)
      tokens = mname.split("-")
      filename = tokens[0]
 
      try:
        # Some PDBs are missing atoms so make sure we have some :/
        # This code is a bit rough but might work
        # We perform a lot of checks here to see if the PDB is valid
        (res, seq) = self._check(model)   
        if not res:
          continue

        # We make a call to get the mmCIF file which we need to make
        # this work
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(filename.upper())
        dn = "" + filename[1] + filename[2] 
        resolution = -1.0
        rvalue = -1.0
        rfree = -1.0
 
        with open(dn + "/" + filename + ".cif", 'r') as f:
          pdbraw = f.readlines()  
          try:
            for line in pdbraw:
              if "_refine.ls_R_factor_R_work " in line:
                tokens = line.split(" ")
                rvalue = float(tokens[-2])

              elif "_refine.ls_R_factor_R_free " in line:
                tokens = line.split(" ")
                rfree=float(tokens[-2])
       
              elif "_reflns.d_resolution_high " in line:
                tokens = line.split(" ")
                resolution=float(tokens[-2])
          except Exception as e:
            print("Error parsing R values", e)
            continue

        if self.conn != None:
          mm = self.conndb.models
          # We search by sequence as we don't want any duplicates
          tt = mm.find_one({"seq" : seq})

          if tt == None:
            if self.conn != None:
              new_model = { "code" : mname, "filename" : bn,\
                  "rfree": rfree, "rvalue": rvalue,\
                  "resolution" : resolutionm, "seq" : seq}}
              mm_id = mm.insert_one(new_model).inserted_id
       
            for atom in model.get_atoms():
              #import pdb; pdb.set_trace()
              #print(atom, dir(atom))
              res = atom.get_parent()
              chain = res.get_parent()
              aserial = int(atom.get_serial_number())
              aname = atom.get_name()
              aaltloc = atom.get_altloc()
              aresname = res.resname
              achainid = chain.id
              aresseq = int(res.id[1])
              ax = float(atom.get_coord()[0])
              ay = float(atom.get_coord()[1])
              az = float(atom.get_coord()[2])
              aocc = float(atom.get_occupancy())
              abfac = float(atom.get_bfactor())
              aele = atom.element
        
              if self.conn != None:
                new_atom = { "model" : mname, "serial" : aserial,\
                    "name": aname, "altloc" : aaltloc,\
                    "resname" : aresname, "chainid" : achainid,\
                    "resseq" : aresseq, "x" : ax, "y" : ay,\
                    "z" : az, "occupancy" : aocc,\
                    "bfactor" : abfac, "element" : aele } 
                mm = self.conndb.atoms
                mm_id = mm.insert_one(new_atom).inserted_id 
              else: 
                print(mname, "atom:", aele, ",", aserial)
          except Exception as e:
            print ("Exception in model insert: ", e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process arguments")
  parser.add_argument('db_name', metavar='db_name',\
      help='The destination database name')
  parser.add_argument('dir_name', metavar='dir_name',\
      help='The directory with the PDBs we want to process.')
  parser.add_argument('--dry-run', dest='dry_run',\
      action='store_true', default=False,\
      help='Do not enter new data into the db. Default False')

  args = vars(parser.parse_args())
  p = PDBMongo()
  print(args)
  if not args['dry_run']:
    p.db_connect(args["db_name"])

  for dirname, dirnames, filenames in os.walk(args["dir_name"]):
    for filename in filenames:
      p.process_pdb(os.path.join(dirname, filename))

