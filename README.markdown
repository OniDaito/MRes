# Contents

## Final networks


## Data generation


## Early networks

This directory contains the early networks we experimented with. They all share code from the common directory. Networks 02, 06, 13 and 23 are included, as these are the ones referred to in the thesis. 

### Running the networks

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
