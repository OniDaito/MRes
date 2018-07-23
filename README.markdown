# Contents

This repository contains the main code for my MRes thesis *Improving CDR-H3 Modelling in Antibodies*. It is split into three sections:

* Early jupyter notebooks
* Early networks in Tensorflow
* Final network in Tensorflow

## Requirements

The following are required to run the networks:

* Postgresql or Mongodb with the antibody data
* [psycopg2](http://initd.org/psycopg/)
* [pymongo](https://github.com/mongodb/mongo-python-driver)
* [tensorflow](https://www.tensorflow.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* Python3

With these installed, you should be all set to go.

## Final networks

The final_network directory contains a set of python files to train both an LSTM and labelling network on a dataset. The *common* directory contains batcher.py which performs conversion of the data from the database to useable vectors for training. This directory also contains several functions common throughout.

The *stats* directory contains programs to generate statistics on the produced data.

The *lstm* directory contains the graph.py definition for the LSTM based network.

To start running the various networks, look into the *run* directory. The file session.py is the entrypoint. To train an LSTM network from scratch, run:

    python session.py

See the bottom of the session.py file for all of the options one can pass to this program.

## Data generation

We use two datasets throughout this work: [AbDb](http://www.bioinf.org.uk/abs/abdb/) and *LoopDB*. I decided to put AbDb into a postgresql database, whereas loopdb ended up in a mongodb. In actuality, it makes little difference which is used.

The table layout in postgresql is as follows:

Atom holds all the atoms for a particular model (not just the CDR-H3). We generate our angles from this. We read the PDB file, setting all these values directly.

                                                            Table "public.atom"
       Column   |         Type         | Collation | Nullable |             Default              | Storage  | Stats target | Description 
    ------------+----------------------+-----------+----------+----------------------------------+----------+--------------+-------------
     model      | character(10)        |           |          |                                  | extended |              | 
     serial     | integer              |           |          |                                  | plain    |              | 
     name       | character varying(4) |           |          |                                  | extended |              | 
     altloc     | character varying(1) |           |          |                                  | extended |              | 
     resname    | character varying(3) |           |          |                                  | extended |              | 
     chainid    | character varying(1) |           |          |                                  | extended |              | 
     resseq     | integer              |           |          |                                  | plain    |              | 
     icode      | character varying(1) |           |          |                                  | extended |              | 
     x          | double precision     |           |          |                                  | plain    |              | 
     y          | double precision     |           |          |                                  | plain    |              | 
     z          | double precision     |           |          |                                  | plain    |              | 
     occupancy  | double precision     |           |          |                                  | plain    |              | 
     tempfactor | double precision     |           |          |                                  | plain    |              | 
     element    | character varying(2) |           |          |                                  | extended |              | 
     charge     | character varying(2) |           |          |                                  | extended |              | 
     id         | integer              |           | not null | nextval('atom_id_seq'::regclass) | plain    |              | 
    Indexes:
        "atom_chainid_idx" btree (chainid)
        "atom_serial_idx" btree (serial)
    Foreign-key constraints:
        "atom_model_fkey" FOREIGN KEY (model) REFERENCES model(code)


Angle lists the phi and psi angles for the CDR H3 loop only. It matches the residue table, with twice the number of entries found in that table.

                                                            Table "public.angle"
      Column  |         Type          | Collation | Nullable |              Default              | Storage  | Stats target | Description 
    ----------+-----------------------+-----------+----------+-----------------------------------+----------+--------------+-------------
     model    | character varying(10) |           |          |                                   | extended |              | 
     phi      | double precision      |           |          |                                   | plain    |              | 
     psi      | double precision      |           |          |                                   | plain    |              | 
     omega    | double precision      |           |          |                                   | plain    |              | 
     resorder | integer               |           |          |                                   | plain    |              | 
     id       | integer               |           | not null | nextval('angle_id_seq'::regclass) | plain    |              | 
    Foreign-key constraints:
        "angle_model_fkey" FOREIGN KEY (model) REFERENCES model(code)

This is the master table. Code is the PDB code, such as 1XT1_1 for example. The filename can be somewhat longer. The endpoint_distance is calculated and added. Accuracy, rvalue and rfree come from the PDB.

                                                               Table "public.model"
        Column     |         Type          | Collation | Nullable |              Default              | Storage  | Stats target | Description 
    ---------------+-----------------------+-----------+----------+-----------------------------------+----------+--------------+-------------
     code          | character(10)         |           | not null |                                   | extended |              | 
     filename      | character varying(40) |           | not null |                                   | extended |              | 
     id            | integer               |           | not null | nextval('model_id_seq'::regclass) | plain    |              | 
     endpoint_dist | double precision      |           |          |                                   | plain    |              | 
     accuracy      | double precision      |           |          |                                   | plain    |              | 
     rvalue        | double precision      |           |          |                                   | plain    |              | 
     rfree         | double precision      |           |          |                                   | plain    |              | 
    Indexes:
        "firstkey" PRIMARY KEY, btree (code)
    Referenced by:
        TABLE "angle" CONSTRAINT "angle_model_fkey" FOREIGN KEY (model) REFERENCES model(code)
        TABLE "atom" CONSTRAINT "atom_model_fkey" FOREIGN KEY (model) REFERENCES model(code)
        TABLE "nerf" CONSTRAINT "nerf_model_fkey" FOREIGN KEY (model) REFERENCES model(code)
        TABLE "redundancy" CONSTRAINT "redundancy_model_fkey" FOREIGN KEY (model) REFERENCES model(code)
        TABLE "redundancy" CONSTRAINT "redundancy_sameas_fkey" FOREIGN KEY (sameas) REFERENCES model(code)
        TABLE "residue" CONSTRAINT "residue_model_fkey" FOREIGN KEY (model) REFERENCES model(code)

The redundancy model lists pairs that are the same. So if a one model is the same as three other models, the model field will contain this model three times, with the model it matches listed in the sameas column.


                                            Table "public.redundancy"
     Column |         Type          | Collation | Nullable | Default | Storage  | Stats target | Description 
    --------+-----------------------+-----------+----------+---------+----------+--------------+-------------
     model  | character varying(10) |           |          |         | extended |              | 
     sameas | character varying(10) |           |          |         | extended |              | 
     Foreign-key constraints:
        "redundancy_model_fkey" FOREIGN KEY (model) REFERENCES model(code)
        "redundancy_sameas_fkey" FOREIGN KEY (sameas) REFERENCES model(code)

The residue table holds the order, label and residue 3 letter abbreviation for the CDR loop in question.

                                                        Table "public.residue"
      Column  |         Type          | Collation | Nullable |               Default               | Storage  | Stats target | Description 
    ----------+-----------------------+-----------+----------+-------------------------------------+----------+--------------+-------------
     model    | character varying(10) |           |          |                                     | extended |              | 
     residue  | character varying(3)  |           |          |                                     | extended |              | 
     reslabel | integer               |           |          |                                     | plain    |              | 
     resorder | integer               |           |          |                                     | plain    |              | 
     id       | integer               |           | not null | nextval('residue_id_seq'::regclass) | plain    |              | 
    Foreign-key constraints:
        "residue_model_fkey" FOREIGN KEY (model) REFERENCES model(code)

The mongo db documents look somewhat similar:

    db.getCollectionNames()
    [
	   "angles",
	   "atoms",
	   "distinctmodels",
	   "models",
	   "residues",
	   "sequence",
	   "smallmodels",
 	   "uniquesequence"
    ]

This is a large dataset so we need to do a lot of pre-processing. Tables *smallmodels*, *uniquesequence* and *distinctmodels* are built from the other tables to make lookups quicker.

The angles table holds the angles for the CDR-H3 only.

    db.angles.findOne();
    {
	    "_id" : ObjectId("5a131b89316131e80959687d"),
	    "model" : "1kya-D255-D282-22",
	    "phi" : 0,
	    "psi" : 133.76422119140625,
	    "omega" : 177.2227783203125,
	    "resorder" : 0
    }


Similar to the postgres table, atoms holds all the atom details from the PDB.
    
    db.atoms.findOne();
    {
	    "_id" : ObjectId("5a09e54c1a1b652159be58e3"),
	    "model" : "1kya-D255-D282-22",
	    "serial" : 13206,
	    "name" : "N",
      "altloc" : " ",
      "resname" : "ASP",
	    "chainid" : "D",
	    "resseq" : 255,
	    "x" : 28.857999801635742,
	    "y" : 53.08700180053711,
	    "z" : 88.5459976196289,
	    "occupancy" : 1,
	    "tempfactor" : 0,
	    "element" : "N",
	    "charge" : "-"
    }

Distinct models lists all these models that are distinct from one another by sequence.

    db.distinctmodels.findOne();
    { "_id" : ObjectId("5a1ababbcc7d4e6b1a9d08e7"), "code" : "10gs-A54-A64-5" }

Models is the main collection.

    db.models.findOne();
    {
	    "_id" : ObjectId("5a09e54b1a1b652159be58e2"),
	    "code" : "1kya-D255-D282-22",
	    "filename" : [
		    "1kya-D255-D282-22"
	    ],
	    "rfree" : 0.276,
	    "rvalue" : 0.253,
	    "resolution" : 2.4
    }

Residues list the three letter abbreviation, the label from the PDB and the order.

    db.residues.findOne();
    {
	    "_id" : ObjectId("5a131b88316131e809596857"),
	    "model" : "1kya-D255-D282-22",
	    "residue" : "TRP",
	    "reslabel" : 258,
	    "resorder" : NumberLong(0)
    }

The sequence lits the one letter sequence for the entire model. We use this table to generate the distinct models.

    db.sequence.findOne();
    {
	    "_id" : ObjectId("5b1a65de1a1b6556fa951d6d"),
	    "code" : "1kya-D255-D282-22",
	    "seq" : "DNYWIRANPNFGNVGFTGGINSAILRYD"
    }

Small models lists the models that are under a specific size (usually 33 residues).

    db.smallmodels.findOne();
    { "_id" : ObjectId("5a1ababbcc7d4e6b1a9d08e7"), "code" : "10gs-A54-A64-5" }

Unique sequence holds the sequence and model of all the unique models that are also small. It is from this table we draw our codes for our datasets.

    db.uniquesequence.findOne();
    {
	   "_id" : ObjectId("5b1a65de1a1b6556fa951d6d"),
	   "code" : "1kya-D255-D282-22",
	   "seq" : "DNYWIRANPNFGNVGFTGGINSAILRYD"
    }


## Early networks

This directory contains the early networks we experimented with. They all share code from the common directory. Networks 02, 06, 13 and 23 are included, as these are the ones referred to in the thesis. 

### Running the networks

To run the networks go to the directory in question and type:

    python nn02.py


## Jupyter Notebooks

The following are the various Jupyter notebooks for some of the key, early concepts I used. They should run fine with the sample data, but are really for understanding the basic concepts only.

### LSTM First Bash for Torsion Angles.ipynb

This notebook goes a little further. It is my first attempt at an LSTM with dropout added.

### TDNN Approximation for Backbone Torsion Angles.ipynb

As the name suggests, this is my attempt at creating a TDNN that attempts to predict backbone torsion angles from the CDR-H3 residues.

## Supporting functions

The **common** directory contains some supporting Python for the grabbing and wrangling of the data for our neural network. For most users, it's enough to know it spits out numpy arrays for our train, test and validate sets.

## Useful Resources

Some useful links for these wanting to get started with neural networks related to structural biology.

* [https://www.tensorflow.org/get_started/mnist/beginners](https://www.tensorflow.org/get_started/mnist/beginners)
* [https://www.tensorflow.org/get_started/mnist/pros](https://www.tensorflow.org/get_started/mnist/pros)
* [https://www.quora.com/Is-a-TDNN-Time-Delay-Neural-Network-same-as-a-1-d-CNN-Convolutional-Neural-Net](A page on Quora about TDNN and CNN) 
* Tensorflow for Machine Intelligence : A Hands-on Introduction to learning algorithms
* Phoneme Recognition Using Time-Delay Neural Networks
* [https://en.wikipedia.org/wiki/Time_delay_neural_network](https://en.wikipedia.org/wiki/Time_delay_neural_network)
* A time delay neural network architecture for efficient modeling of long temporal contexts
* Computational Intelligence : An Introduction
