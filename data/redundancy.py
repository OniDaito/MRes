"""
redundancy.py - Read the redundancy file from AbDb into Postgres
author : Benjamin Blundell
email : me@benjamin.computer

"""
import sys, time, psycopg2, traceback, os, argparse

def parse_file(filepath, dbname):
  pairs = []
  tt = 0
  with open(filepath, 'r') as f:
    for line in f.readlines():
      line = line.replace(" ","")
      line = line.replace("\n","")
      if line[-1] == ",":
        line = line[:-1]
      tokens = line.split(",")
      tt += len(tokens)
      if len(tokens) > 1:
        ta = tokens[0]
        for same in tokens[1:]:
          pairs.append((ta,same))

  print("Total entries:",tt)
  conn = psycopg2.connect("dbname=" + dbname + " user=postgres")
  cur = conn.cursor()

  for pair in pairs: 
    cur.execute("SELECT * FROM model where code = '" + pair[0] + "'")
    models0 = cur.fetchall()
    cur.execute("SELECT * FROM model where code = '" + pair[1] + "'")
    models1 = cur.fetchall()

    if len(models0) > 0 and len(models1) > 0:
      cur.execute("INSERT INTO redundancy (model, sameas) VALUES (%s, %s)", (pair[0], pair[1]))
      conn.commit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process arguments")
  parser.add_argument('file', metavar='file_name', help='The file of redundancy information.')

  args = vars(parser.parse_args())
  if args['file']:
    parse_file(args['file'], "pdb_martin")

 
