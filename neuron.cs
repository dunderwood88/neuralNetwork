using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public class Neuron{

        private List<double> _weights;
        private double _delta;
        private double _output;
        private double _bias;
        private double _lambda = 6;
        private double _learnRate = 0.5;

        //default constructor: only called for input neurons
        public Neuron(){
            _weights = new List<double>();
            _weights.Add(1.0);
        }
        
        //overload constructor for all other layers
        public Neuron(int numberInputs){

            //Set up tracking of the input weights
            //Begin by assigning random weights
            
            _weights = new List<double>();

            for(int i = 0; i < numberInputs; i++){

                _weights.Add(Util.GetRandom() * 2 - 1);
            }

        }

        public void FeedForward(List<double> inputs, bool isInput){

            double sum = 0.0;

            for(int i = 0; i < inputs.Count; i++){

                sum += inputs[i] * _weights[i]; 
            }
            //Console.WriteLine(sum);

            if(!isInput){
                _output = TransferFunction(sum);
            }
            else{
                _output = sum;
            }
            
        }

        
        //Could possibly generalise this using a delegate?
        private double TransferFunction(double input){

            return 1 / (1 + Math.Exp(-_lambda * (input + _bias)));

        }

        //Could possibly generalise this using a delegate?
        private double TransferFunctionDerivative(double x){

            return TransferFunction(x) * (1 - TransferFunction(x));

        }


        public double Output(){
            
            return _output;
        }

        public void ComputeDelta(double value){

            _delta = _output * (1 - _output) * value;
        }

        public double WeightedDelta(int idx){

            return _delta * _weights[idx];
        }


        public void AdjustWeights(List<double> input){

            for(int i = 0; i < _weights.Count; i++){

                _weights[i] += input[i] * _delta * _learnRate;

            }

            _bias += _delta * _learnRate;
            
            
        }


    }

}