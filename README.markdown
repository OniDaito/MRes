A repository for my latest work on the MRes, fit for showing to the wider world.

# Contents

## net01 - nn01/nn01.py

Run with Tensorflow on a gpu with:

    python nn01.py

Load a pre-trained version:

    python nn01.py -r

First stab at an actual neural net for dealing with the problem. My approach is to use a convolutional neural net design and *shoe-horn* it into being a TDNN. 

In order to do this, I take each bitfield as a separate channel. So where as an image has RGB channels, I have one channel per amino acid. It's like a 1D image with 21 colours. This the data we take into the net to begin with.

Layer one convolves this input with a 1 x 4 kernel along the 1D vector which represents the amino acid polypeptide (some of which will be zero and should not be counted, but I don't know how to do this yet).

The depth of the convolution should be 21, the number of Amino acids. I'd like each neuron in this layer to be sensitive to one of the amino-acids over the course of the time. The output size *should* be the same as the input size with the 1 step stride and 'SAME' padding but I'm really not sure about this bit.

The next layer, I hope, is fully connected. There follows a reshape and then a final output layer which is 4 * max_cdr_length long it would appear. 

Things that make this not work are the CDRs less than max length long. We need to figure out how to drop these from the training data when they appear somehow?

Im using tanh as activation functions as spans the range I'm looking for (-1 to 1). Not sure if thats needed internally however. It doesn't seem to affect the quality of the output.

In addition, changing window size doesn't change much but reducing the batch size to 1 helps the reported accuracy a bit.

### Diagrams of the output thus far from the tensorboard and dia

This needs more work I think - need to output the various sizes of the tensors.

![dia version](nn01/nn01.png)

Use: 
    tensorboard --logdir=./summaries_01/train/

![tensorboard](nn01/nn01_tensorboard.png)


### Things I've read at this point

* [https://www.tensorflow.org/get_started/mnist/beginners](https://www.tensorflow.org/get_started/mnist/beginners)
* [https://www.tensorflow.org/get_started/mnist/pros](https://www.tensorflow.org/get_started/mnist/pros)
* [https://www.quora.com/Is-a-TDNN-Time-Delay-Neural-Network-same-as-a-1-d-CNN-Convolutional-Neural-Net](A page on Quora about TDNN and CNN) 
* Tensorflow for Machine Intelligence : A Hands-on Introduction to learning algorithms
* Phoneme Recognition Using Time-Delay Neural Networks
* [https://en.wikipedia.org/wiki/Time_delay_neural_network](https://en.wikipedia.org/wiki/Time_delay_neural_network)
* A time delay neural network architecture for efficient modeling of long temporal contexts
* Computational Intelligence : An Introduction
