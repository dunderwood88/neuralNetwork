using System.Collections.Generic;
using NeuralNetworks;

namespace ConsoleApplication{

    public class Program{

        public static void Main(string[] args){
            
            FeedForwardNet network = new FeedForwardNet(new int[]{2,2,1});
            //FeedForwardNet network = new FeedForwardNet(new int[]{4,7,3});
           
            List<Pattern> patterns = new List<Pattern>();
            List<double[]> inputs = new List<double[]>();
            List<double[]> expectedValues = new List<double[]>();


            //Load patterns
            //patterns = Util.LoadPatterns("../../Desktop/pima-indians-diabetes.data.csv",8);
            patterns = Util.LoadPatterns("Patterns.csv",2);
            //patterns = Util.LoadPatterns("iris.data.csv",4);

            foreach(Pattern p in patterns){
                inputs.Add(p.Inputs());
                expectedValues.Add(p.Outputs());
            }

            network.Learn(inputs, expectedValues);
            

            
        }
    }
}