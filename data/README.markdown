# Dataset Creation

Our data come from two sources, the [Protein Databank]() (referred to as LoopDB) and [AbDb]().

## LoopDB 

To recreat LoopDB one must follow the following process:

1. install the required software
2. create a mirror of the PDB in it's entiriety
3. run the script that selects appropriate pdb files from this list
4. run the script to digest PDB files into a [mongo database]().

### Required software

There are a few software requirements:

* python3
* [biopython]()
* [bioptools](https://github.com/ACRMGroup/bioptools)
* [pymongo]()
* GCC
* Git
* A working [mongodb]() install.

You can install these using programs such as '''pip''' using either your base install or with [virtualenv](). For example:

    pip install pymongo

Bioptools is a useful set of small, C programs that work well with PDB files. You can extract zones, chains, headers, perform stats and do a lot of cleaning. We use it here to extract the loop sections from the PDB.

Follow the install instructions for bioptools at [https://github.com/ACRMGroup/bioptools/blob/master/INSTALL.md](https://github.com/ACRMGroup/bioptools/blob/master/INSTALL.md).

### Mirroring the PDB

The first script is called [FTPMirror](https://github.com/AndrewCRMartin/bioscripts/tree/master/ftpmirror). Instructions can be found in that github repository. It is written in Perl and is straightforward to follow. It relies on a config file in the same directory as the script. I have included the config file I used in this repository.

### Building LoopDB

You'll need a fair bit of disk space and time to build LoopDB. We need to read a lot of PDB files, find out if they have loop structures, extract these structures, process these resulting files some more in case of errors and then write it all to a database for easy batching. Fortunately, we only have to do this once.

The program '''loopdb''' works on Linux and can be built by running the make file with the command

    make

It relies on having a working GCC install. It's role is to find loop-like structures from all the PDB files you've mirrored in the previous step. 

Several programs are built by this process, but the most relevant are '''scanloopdb''' and '''buildloopdb'''. The file distances.h is also useful. It contains the distances and standard deviations we use as criteria to find our antibody-loop-like structures. The defaults are generally acceptable but it's worth looking at.

Run the following command with the PDB mirror directory:

    ./buildloopdb -m 5 -x 28 <path to pdb mirror dir> out.db

Once the lookup file is built, we can use it to look for loops in a PDB file. Rather than just the one, lets use a little bash script to generate a summary of the entire databank:

    for i in `ls <PATH TO PDBS>`; do ./scanloopdb out.db <PATH TO PDBS>/$i >> summary.txt; done
 
'''scanloopdb''' is run against each PDB to see if there are any loop structures that are useful to us. These are then written to the file '''summary.txt'''. This summary.txt is probably quite large. It lists the areas within the PDB that are acceptable as loops along with the score.

We need to extract the loops and create a new set of PDB files. We have a script for that:

    ./extract.sh <path to summary file> <path to PDB Mirror> <path to output files>

This results in a set of PDB loop files. The final step is to load these into the mongodb ready for processing by our neural network. To do that, we have one last python program:


    python pdb_to_mongo.py <dbname> <path to pdbs>

Again, this will take quite a while. On our base system this took a few days. Processing the files includes generating angles from atom positions, checking residues are correct and downloading the accuracy scores for each file.

### Building AbDb

AbDb is much easier to build than LoopDB. The files are already processed and numbered. 
