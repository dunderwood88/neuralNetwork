using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetworks{

    public static class Util{

        private static Random rnd = new Random();
        private static List<Pattern> _patterns;
        
        public static double GetRandom()
        {
            return rnd.NextDouble();

        }

        public static List<Pattern> LoadPatterns(string filename)
        {
            _patterns = new List<Pattern>();
            StreamReader file = File.OpenText(filename);
            while (!file.EndOfStream)
            {
                string line = file.ReadLine();
                _patterns.Add(new Pattern(line));
            }
            file.Dispose();

            return _patterns;
        }

    }

}