using System;
using System.Collections.Generic;
using NeuralNetworks;

namespace ConsoleApplication{

    public class Program{

        public static void Main(string[] args){
            
            Net network = new Net(new int[]{2,3,1});

            List<double> inputs = new List<double>();

            inputs.Add(1.2);
            inputs.Add(1.6);

            network.FeedForward(inputs);

            
        }
    }
}