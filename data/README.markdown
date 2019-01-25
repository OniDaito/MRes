# Dataset Creation

Our data come from two sources, the [Protein Databank](https://www.wwpdb.org/) (referred to as LoopDB) and [AbDb](http://www.bioinf.org.uk/abs/abdb/).

Both of these sources should be processed into a postgresql database. From there, we can create datasets of various kinds, as python pickle files, which we can then use with our neural networks.


## Required software

There are a few software requirements:

* python3
* [biopython](https://biopython.org/) 
* [bioptools](https://github.com/ACRMGroup/bioptools)
* [psycopg2](http://initd.org/psycopg/)
* GCC
* Git
* A working postgresql install.

You can install these using programs such as *pip* using either your base install or with [virtualenv](https://virtualenv.pypa.io/en/stable/). For example:

    pip install psycopg2

Bioptools is a useful set of small, C programs that work well with PDB files. You can extract zones, chains, headers, perform stats and do a lot of cleaning. We use it here to extract the loop sections from the PDB.

Follow the install instructions for bioptools at [https://github.com/ACRMGroup/bioptools/blob/master/INSTALL.md](https://github.com/ACRMGroup/bioptools/blob/master/INSTALL.md).

## Postgresql setup

Create a database using whatever method you normally use. Perhaps something like this:

    psql -U postgres
    create database loopdb
    \q

Then you can import the schema from this repository:

    psql -U postgres -d loopdb < schema.sql

I generally keep AbDb and LoopDB as seperate databases, though you can indeed merge them together.

## LoopDB 

To recreate LoopDB one must follow the following process:

1. Install the required software
2. Create a mirror of the PDB in it's entiriety
3. Run the script that selects appropriate pdb files from this list
4. Run the script to digest PDB files into a Postgresql database

### Mirroring the PDB

The first script is called [FTPMirror](https://github.com/AndrewCRMartin/bioscripts/tree/master/ftpmirror). Instructions can be found in that github repository. It is written in Perl and is straightforward to follow. It relies on a config file in the same directory as the script. I have included the config file I used in this repository.

### Building LoopDB

You'll need a fair bit of disk space and time to build LoopDB. We need to read a lot of PDB files, find out if they have loop structures, extract these structures, process these resulting files some more in case of errors and then write it all to a database for easy batching. Fortunately, we only have to do this once.

The program *loopdb* works on Linux and can be built by running the make file with the command

    make

It relies on having a working GCC install. It's role is to find loop-like structures from all the PDB files you've mirrored in the previous step. 

Several programs are built by this process, but the most relevant are *scanloopdb* and *buildloopdb*. The file distances.h is also useful. It contains the distances and standard deviations we use as criteria to find our antibody-loop-like structures. The defaults are generally acceptable but it's worth looking at.

Run the following command with the PDB mirror directory:

    ./buildloopdb -m 5 -x 28 <path to pdb mirror dir> out.db

Once the lookup file is built, we can use it to look for loops in a PDB file. Rather than just the one, lets use a little bash script to generate a summary of the entire databank:

    for i in `ls <PATH TO PDBS>`; do ./scanloopdb out.db <PATH TO PDBS>/$i >> summary.txt; done
 
*scanloopdb* is run against each PDB to see if there are any loop structures that are useful to us. These are then written to the file *summary.txt*. This summary.txt is probably quite large. It lists the areas within the PDB that are acceptable as loops along with the score.

We need to extract the loops and create a new set of PDB files. We have a script for that:

    ./extract.sh <path to summary file> <path to PDB Mirror> <path to output files>

This results in a set of PDB loop files. The final step is to load these into the mongodb ready for processing by our neural network. To do that, we have one last python program:


    python pdb_to_postgres.py <dbname> <path to pdbs>

Again, this will take quite a while. On our base system this took a few days. Processing the files includes generating angles from atom positions, checking residues are correct and downloading the accuracy scores for each file.

## AbDb

AbDb is much easier to build than LoopDB. The files are already processed and numbered. There are a number of redundancies however and these are provided in a text file you can download from here: [http://www.bioinf.org.uk/abs/abdb/](http://www.bioinf.org.uk/abs/abdb/). Once you have the PDB files in a particular directory, all you need to do is run the following command:

    python pdb_to_postgres.py --rfile <path to redundancy file> <dbname> <path to pdbs>

## Generating a pickle file for final use

To generate the pickle file, you can just run the neural network and it will generate one, but I've found it better to write a generator with more options, where one can choose various options, such as loop length, representation type and similar. The file *batcher.py* in the *final_network/common/* will generate the pickle file from a particular database. Several options exist to limit various sizes, use different encoding schemes and such, but the minimum needed is as follows:

    python batcher.py --dbname loopdb --out test.pickle --max 28

*dbname* refers to the name of the postgresql database we are reading from. *out* is the name of the picklefile and *max* refers to the maximum length of loop to be included.

