using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public class Neuron{

        private double _bias;
        private List<double> _weights;
        private double _weightSum;
        private double _output = double.MinValue;

        public Neuron(int numberInputs){

            //Set up tracking of the input weights
            //Begin by assigning random weights
            
            _weights = new List<double>();
            _weightSum = 0;

            for(int i = 0; i < numberInputs; i++){

                _weights.Add(Util.GetRandom() * 2 - 1);
            }

        }

        public void FeedForward(List<double> inputs){

            double sum = 0.0;

            for(int i = 0; i < inputs.Count; i++){
                
                Console.WriteLine("input " + (i+1).ToString() + " = " + inputs[i].ToString());

                sum += inputs[i] * _weights[i];

                //Console.WriteLine("sum = " + sum.ToString());
                
                _output = TransferFunction(sum);
                
            }
            
            
        }

        
        //Could possibly generalise this using a delegate?
        private double TransferFunction(double input){

            return 1 / (1 + Math.Exp(input));

        }

        //Could possibly generalise this using a delegate?
        private double TransferFunctionDerivative(double x){

            return TransferFunction(x) * (1 - TransferFunction(x));

        }

        public double Output(){
            
            return _output;
        }

    }

}