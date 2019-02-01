# Contents

This repository contains the main code for my MRes thesis *Improving CDR-H3 Modelling in Antibodies*. It is split into three sections:

* Early jupyter notebooks
* Early networks in Tensorflow
* Final network in Tensorflow

The final network is the most important, whereas the other two areas show basic examples and evolution of the network proper.

The published results, descriptions of the problem and the entire process can be found in my thesis **Improving CDR-H3 modelling in antibodies** which can be downloaded from Zenodo here: [https://zenodo.org/record/2549566](https://zenodo.org/record/2549566)  

## Requirements

The following are required to run the networks:

* Postgresql with the antibody data or a downloaded pickle file
* [psycopg2](http://initd.org/psycopg/)
* [tensorflow](https://www.tensorflow.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* Python3

With these installed, you should be all set to go.

## Quick test

With the software installed, try the following to get a basic network training:

1. Download the LoopDB dataset picklefile - [https://zenodo.org/record/2549426](https://zenodo.org/record/2549426)
2. Go to the *MRes/final_network/run/* directory
3. Run the following command:

    python session.py --picklename abdb.pickle --savepath ./save --maxlength 28

With everything installed, this should start the network training. It'll probably take quite a while. 

## Final networks

The final_network directory contains a set of python files to train both an LSTM and labelling network on a dataset. The *common* directory contains batcher.py which performs conversion of the data from the database to useable vectors for training. This directory also contains several functions common throughout.

The *stats* directory contains programs to generate statistics on the produced data.

The *lstm* directory contains the graph.py definition for the LSTM based network.

To start running the various networks, look into the *run* directory. The file session.py is the entrypoint. To train an LSTM network from scratch, run:

    python session.py

The command I use the most often to generate the networks and data is as follows:

    python session.py --picklename abdb.pickle --savepath ./save --maxlength 28

This command loads the abdb.pickle file (which you can download from Zenodo), saving the results to the save directory, with a maximum loop length of 28.

See the bottom of the session.py file for all of the options one can pass to this program.

## Datasets

We use two datasets throughout this work: [AbDb](http://www.bioinf.org.uk/abs/abdb/) and *LoopDB*. AbDb is a smaller set of loops from antibody pdb models, whereas LoopDB is a collection of loops derived from all the models in the Protein Databank that satisfy a certain criteria for similiarity with CDR loops.

Our neural networks work with a [Python Pickle File](https://wiki.python.org/moin/UsingPickle), which you can either generate yourself or download from Zenodo.

### Download from Zenodo

* The LoopDB set can be downloaded here: [https://zenodo.org/record/2549426](https://zenodo.org/record/2549426) (loopdb.pickle). This contains the latest loops from the protein databank, from length 3 to 28.

### Generate your own pickle file

AbDb can be downloaded from [http://www.bioinf.org.uk/abs/abdb/](http://www.bioinf.org.uk/abs/abdb/). It consists ofa set of PDB files. 

The LoopDB must be downloaded from Protein Databank, file by file. There are a set of C programs you can build in order to do this. More information can be found in the [relevant README.markdown file in the data directory](https://github.com/OniDaito/MRes/blob/master/data/README.markdown).

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
