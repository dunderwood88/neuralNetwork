using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public class Net{

        private List<Layer> _layers;


        public Net(int[] structure){
            
            Console.WriteLine("Creating neural network of " + structure.Length.ToString() + " layers");

            _layers = new List<Layer>();

            for(int i = 0; i < structure.Length; i++){
                
                Console.WriteLine("Layer " + (i + 1).ToString());

                int prevLayerRank;

                //if we are on the first layer, set the rank of the "previous" layer to 1 (i.e. 1 input per neuron)
                if(i == 0){
                    prevLayerRank = 1;
                } 
                else{
                    prevLayerRank = structure[i-1];
                }

                _layers.Add(new Layer(structure[i], prevLayerRank));

            }

        }


        //Initialises the feedforward process
        public void FeedForward(List<double> inputValues){

            Console.WriteLine();
            Console.WriteLine("Feeding forward...");


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

                _layers[0].FeedNeurons(i, inputAsList);
            }   

            //Forward propagation

            for(int layerNum = 1; layerNum < _layers.Count; layerNum++){

                Console.WriteLine("Layer " + (layerNum + 1).ToString());

                for(int i = 0; i < _layers[layerNum].Size(); i++){

                    _layers[layerNum].FeedNeurons(i, _layers[layerNum - 1].Outputs());

                }

            }
            
            //Print the final outputs
            foreach(double o in _layers[_layers.Count - 1].Outputs()){

                Console.WriteLine(o.ToString());

            }

        }

    }

}