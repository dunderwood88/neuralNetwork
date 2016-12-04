using System.Collections.Generic;
using NeuralNetworks;

namespace ConsoleApplication{

    public class Program{

        public static void Main(string[] args){
            
            FeedForwardNet network = new FeedForwardNet(new int[]{2,2,1});
           
            List<Pattern> patterns = new List<Pattern>();
            List<double[]> inputs = new List<double[]>();
            List<double> expectedValues = new List<double>();


            //Load patterns
            //Util.LoadPatterns("../../Desktop/pima-indians-diabetes.data.csv");
            patterns = Util.LoadPatterns("Patterns.csv");

            foreach(Pattern p in patterns){
                inputs.Add(p.Inputs());
                expectedValues.Add(p.Output());
            }

            /*inputs.Add(new double[] {0,0});
            inputs.Add(new double[] {0,1});
            inputs.Add(new double[] {1,0});
            inputs.Add(new double[] {1,1});

            expectedValues.Add(0);
            expectedValues.Add(1);
            expectedValues.Add(1);
            expectedValues.Add(0);*/


            network.Learn(inputs, expectedValues);
            

            
        }
    }
}