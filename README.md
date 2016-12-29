Feed forward backpropagation neural network implementation.
Aiming to make for general use, for a network of arbitrary topology.

Currently, this implementation consists of:

- User-specificification of layer amount, and input/output/hidden layer ranks;
- Read-in of csv-type data: assumes numerical csv data, where each line contains input data followed by expected output(s) (must specify number of columns that are inputs);
- Uses a sigmoid as the transfer function.

NOTE: Forward connections exist between all possible neuron pairs.

FeedForwardNetwork object is constructed with mandatory integer array argument, which specifies the structure of the network, e.g.

FeedForwardNet network = new FeedForwardNet(new int[]{2,2,1});

creates a network with 2 input neurons, 2 hidden neurons, and a single output neuron.

Optional arguments include: 
- learnRate for weight adjustments;
- lambda parameter for Sigmoid function;
- maximum iterations for learning loop;
- threshold error value for convergence. 

Learn method is called with input and expected output arguments passed, each as lists of double arrays.

Still to add:

- Testing and accuracy methods;
- Generalisation of transfer functions;
- Exporting of network parameters to text file;
- Various dependency injections, generalisation and tidying up;
- Optimisation once complete;
- Probably a lot more...
