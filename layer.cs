using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public class Layer{

        private int _rank;
        private List<Neuron> _neurons;

        public Layer(int rank, int prevRank){

            _rank = rank;
            _neurons = new List<Neuron>();

            for(int i = 0; i < rank; i++){

                Console.WriteLine("Adding neuron " + (i + 1).ToString());
                _neurons.Add(new Neuron(prevRank));

            }

        }

        //Assigns input values into i'th neuron
        public void FeedNeurons(int i, List<double> inputs){
            
            _neurons[i].FeedForward(inputs);
        }

        public int Size(){
            return _rank;
        }

        public List<double> Outputs(){
            
            List<double> outputs = new List<double>();
            foreach(Neuron x in _neurons)
                outputs.Add(x.Output()); 

            return outputs;
        }

    }

}