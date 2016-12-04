Feed forward backpropagation neural network implementation.
Aiming to make for general use, for a network of arbitrary topology.

FeedForwardNetwork object is constructed with mandatory integer array argument, which specifies the structure of the network, e.g.
FeedForwardNet network = new FeedForwardNet(new int[]{2,2,1});
creates a network with 2 input neurons, 2 hidden neurons, and a single output neuron.

Learn method is called with input and expected output arguments passed.

Still to add:

- Full functionality for multiple outputs;
- Convergence threshold control;
- Testing and accuracy methods;
- Probably a lot more...