using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks{

    public class Net{

        private List<Layer> _layers;


        public Net(int[] structure){
            
            Console.WriteLine("Creating neural network of " + structure.Length.ToString() + " layers");

            _layers = new List<Layer>();

            for(int i = 0; i < structure.Length; i++){
                
                Console.WriteLine("Layer " + (i + 1).ToString());

                //if we are on the first layer, there is no previous layer to pass rank for
                if(i == 0){
                    _layers.Add(new Layer(structure[i]));
                } 
                else{
                    _layers.Add(new Layer(structure[i], structure[i-1]));
                }

            }

        }

        public void Learn(List<double[]> inputs, List<double> expectedValues){

            for(int i = 0; i < 500; i++){

                Console.WriteLine("Epoch " + (i + 1).ToString());
                
                for(int x = 0; x < inputs.Count; x++){

                    List<double> valueSet = inputs[x].ToList();
                    List<double> expectedVals = new List<double>();
                    expectedVals.Add(expectedValues[x]);

                    FeedForward(valueSet);
                    BackPropagation(expectedVals);
                    
                }
            }
            
        }


        //Initialises the feedforward process
        public void FeedForward(List<double> inputValues){

            //Console.WriteLine();
            //Console.WriteLine("Feeding forward...");


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

                Console.WriteLine("Output: " + o.ToString());

            }

        }

        //Back propagation
        public void BackPropagation(List<double> targetValues){

            //output layer
            //for each neuron in output layer
                //calculate outputValue - targetValue
                //multiply by derivative of activation function to obtain delta value
            _layers[_layers.Count - 1].ComputeDeltas(targetValues);

            //Console.WriteLine("Backprop of layer " + _layers.Count + " done");

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
                    
                    //Console.WriteLine("Backprop of layer " + (i + 1).ToString() + " done");
            }
            
            //repeat up to input layer

            //Adjust weights
            for(int i = _layers.Count - 1; i > 0; i--){

                _layers[i].AdjustLayerWeights(_layers[i - 1].Outputs());

            }

        }


    }

}