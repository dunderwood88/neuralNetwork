using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks{

    public class FeedForwardNet{

        private int _maxIterations;
        private int _iteration;
        private int _size;
        private double _error;
        private List<Layer> _layers;

        //Various overloads for the constructor depending on specified inputs, default if missing
        public FeedForwardNet(int[] structure){
            NetSetup(structure, 0.5, 5, 2000, 0.01);
        }

        public FeedForwardNet(int[] structure, double learnRate){
            NetSetup(structure, learnRate, 5, 2000, 0.01);
        }

        public FeedForwardNet(int[] structure, double learnRate, double lambda){
            NetSetup(structure, learnRate, lambda, 2000, 0.01);
        }

        public FeedForwardNet(int[] structure, double learnRate, double lambda, int iter){
            NetSetup(structure, learnRate, lambda, iter, 0.01);
        }

        public FeedForwardNet(int[] structure, double learnRate, double lambda, int iter, double thErr){
            NetSetup(structure, learnRate, lambda, iter, thErr);
        }



        public void NetSetup(int[] structure, int iter, double learnRate, double lambda, double thErr){
            
            Console.WriteLine("Creating feed forward neural network of " + structure.Length.ToString() + " layers");

            _layers = new List<Layer>();
            _maxIterations = iter;
            _error = thErr;
            _size = structure.Length;

            for(int i = 0; i < structure.Length; i++){
                
                Console.WriteLine("Layer " + (i + 1).ToString());

                //if we are on the first layer, there is no previous layer to pass rank for
                if(i == 0){
                    _layers.Add(new Layer(structure[i]));
                } 
                else{
                    _layers.Add(new Layer(structure[i], structure[i-1]));
                }

                _layers[i].SetParams(learnRate, lambda);

            }

        }


        public void Learn(List<double[]> inputs, List<double[]> expectedValues){

            double errLoop;

            do{

                errLoop = 0;
                Console.WriteLine("Epoch " + (_iteration + 1).ToString());                
                for(int x = 0; x < inputs.Count; x++){

                    List<double> valueSet = inputs[x].ToList();
                    List<double> expectedVals = new List<double>();
                    double err = 0;

                    foreach(double e in expectedValues[x]){
                        expectedVals.Add(e);
                    }

                    FeedForward(valueSet);
                    BackPropagation(expectedVals);

                    //Feed back the total error
                    for(int z = 0; z < expectedVals.Count; z++){
                        err += Math.Pow((expectedVals[z] - _layers[_size - 1].Outputs()[z]),2);
                    }
                    errLoop += err;

                }

                Console.WriteLine("Error:" + errLoop);

                _iteration++;
                if(_iteration > _maxIterations - 1){
                    Console.WriteLine("Could not converge, try again");
                    Reinitialise();
                    _iteration = 0;

                }
            }while (errLoop > _error);

            Console.WriteLine("Converged with a total error of " + errLoop);
        }

        private void Reinitialise(){

            Console.WriteLine("Reinitialising...");

            for(int x = 1; x <_layers.Count; x++){
                _layers[x].Reinitialise();
                //_layers[x].SetParams(learnRate, lambda);
            }
        }


        //Initialises the feedforward process
        private void FeedForward(List<double> inputValues){

            if(inputValues.Count != _layers[0].Size()){
                Console.WriteLine("The number of input params does not equal the number of neurons");
                return;
            }

            //Assign the input values into the input neurons
            for(int i = 0; i < inputValues.Count; i++){
                
                /*Set up a List with a single value, as the Neuron FeedForward method 
                takes a List argument (for re-usability)*/
                List<double> inputAsList = new List<double>(); 
                inputAsList.Add(inputValues[i]);

                _layers[0].FeedNeurons(i, inputAsList, true);
            }   

            //Forward propagation
            for(int layerNum = 1; layerNum < _layers.Count; layerNum++){

                for(int i = 0; i < _layers[layerNum].Size(); i++){

                    _layers[layerNum].FeedNeurons(i, _layers[layerNum - 1].Outputs(), false);

                }

            }
            
            //Print the final outputs
            foreach(double o in _layers[_layers.Count - 1].Outputs()){

                //Console.WriteLine("Output: " + o.ToString());
            }

        }

        //Back propagation
        private void BackPropagation(List<double> targetValues){

            //output layer
            //for each neuron in output layer
                //calculate outputValue - targetValue
                //multiply by derivative of activation function to obtain delta value
            _layers[_layers.Count - 1].ComputeDeltas(targetValues);

            //hidden layers
            //for each neuron in hidden layer i
                //sum all of (deltas * weights[i]) of the next layer, where weights[i] is the weight connecting the current neuron
                //to that of the next layer
                //multiply this sum by derivative of activation function to obtain delta value
            for(int i = _layers.Count - 2; i > 0; i--){

                    //for each neuron in the current layer
                    for(int j = 0; j < _layers[i + 1].Size(); j++){
                        _layers[i].ComputeDeltas(_layers[i + 1].SumWeightedDeltas(j));
                    }
            }

            //Adjust weights
            for(int i = _layers.Count - 1; i > 0; i--){
                _layers[i].AdjustLayerWeights(_layers[i - 1].Outputs());
            }
        }


    }

}