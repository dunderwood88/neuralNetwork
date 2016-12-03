using System;
using System.Collections.Generic;
using NeuralNetworks;

namespace ConsoleApplication{

    public class Program{

        public static void Main(string[] args){
            
            Net network = new Net(new int[]{2,2,1});
           
            List<double[]> inputs = new List<double[]>();
            List<double> expectedValues = new List<double>();

            //inputs.Add(new double[] {0,1});
            //expectedValues.Add(1);

            inputs.Add(new double[] {0,0});
            inputs.Add(new double[] {0,1});
            inputs.Add(new double[] {1,0});
            inputs.Add(new double[] {1,1});

            expectedValues.Add(0);
            expectedValues.Add(1);
            expectedValues.Add(1);
            expectedValues.Add(0);


            network.Learn(inputs, expectedValues);
            

            
        }
    }
}