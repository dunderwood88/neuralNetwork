using System;
using System.Collections.Generic;

namespace NeuralNetworks{

    public static class Util{

        private static Random rnd = new Random();
        
        public static double GetRandom()
        {
            return rnd.NextDouble();
        }

    }

}