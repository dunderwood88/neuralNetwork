
namespace NeuralNetworks{

    //Currently only assumes there is one output value listed at the end of each line
    public class Pattern{
        
        private double[] _inputs;
        private double[] _outputs;
     
        public Pattern(string value, int numInputs)
        {
            string[] line = value.Split(',');
            
            _inputs = new double[numInputs];
            _outputs = new double[line.Length - numInputs];
            for (int i = 0; i < numInputs; i++)
            {
                _inputs[i] = double.Parse(line[i]);
            }

            for (int i = 0; i < (line.Length - numInputs); i++)
            {
                _outputs[i] = double.Parse(line[i + numInputs]);
            }
            
        }

        public double[] Inputs(){
            return _inputs;
        }

        public double[] Outputs(){
            return _outputs;
        }
     
    }

}