using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public class Layer{

        private int _rank;
        private List<Neuron> _neurons;
        private List<double> _outputs;

        //constructor called for input layer
        public Layer(int rank){
            
            _rank = rank;
            _neurons = new List<Neuron>();
            _outputs = new List<double>();

            for(int i = 0; i < rank; i++){
                Console.WriteLine("Adding neuron " + (i + 1).ToString());
                _neurons.Add(new Neuron());
            }
        }

        //overload constructor called for all other layers
        public Layer(int rank, int prevRank){

            _rank = rank;
            _neurons = new List<Neuron>();
            _outputs = new List<double>();

            for(int i = 0; i < rank; i++){
                Console.WriteLine("Adding neuron " + (i + 1).ToString());
                _neurons.Add(new Neuron(prevRank));
            }
        }

        //Assigns input values into i'th neuron
        public void FeedNeurons(int i, List<double> inputs, bool isInput){         
            _neurons[i].FeedForward(inputs, isInput);
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

        public void ComputeDeltas(List<double> targetValues){

            for(int i = 0; i < _neurons.Count; i++){
                _neurons[i].ComputeDelta(targetValues[i] - _neurons[i].Output());
            }
        }

        //Overload method for all hidden layers
        public void ComputeDeltas(double summedWeightedDeltas){

            foreach(Neuron neuron in _neurons){
                neuron.ComputeDelta(summedWeightedDeltas);
            }
        }

        public double SumWeightedDeltas(int index){

            double sum = 0.0;

            foreach(Neuron neuron in _neurons){
                sum += neuron.WeightedDelta(index);
            }
            return sum;
        }

        public void AdjustLayerWeights(List<double> inputs){

            for(int i = 0; i < _rank; i++){
                _neurons[i].AdjustWeights(inputs);
            }
        }

        public void Reinitialise(){

            foreach(Neuron n in _neurons){
                n.Reinitialise();
            }
        }

        public void SetParams(double learnRate, double lambda){

            foreach(Neuron n in _neurons)
                n.SetParams(learnRate, lambda);
        }

    }

}