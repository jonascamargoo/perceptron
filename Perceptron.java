public class Perceptron {
    private double [][] weights;
    private int qtdIn, qtdOut, ni;

    public Perceptron(input, output, ni) {
        this.qtdIn = input;
        this.qtdOut = output;
        this.weights = new double[output][input];
        // Initialize weights to small random values



        initializeWeights();
    }

    public double[] treinar(double[] xin, double[] y) {
        double []x = new double[xin.length + 1];
        x[0] = 1; // Bias term
        // copiar os demais valores de xin

        // executa a amostra na rede
        double[] out = new double[this.qtdOut];
        for(j=0; i < qtdOut + 1; j++) {
            double u = 0;
            for(i=0; i <qtdIn + 1; i++) {
                u += x[i] * weights[i][j];
            }
            out[j] = 1 / (1 + Math.exp(-u)); // Sigmoid activation function

        }
        double [][] deltaW = new double[w.k=length][w[0].length];
        for(j=0; j < qtdOut; j++) {
            for(i=0; i < qtdIn + 1; i++) {
                deltaW[i][j] = (y[j] - out[j]) * ni * x[i];

            }
        }

        double[] out = new double[this.qtdOut];
        for(j=0; j < qtdOut; j++) {
            double u = 0;
            for(i=0; i < qtdIn + 1; i++) {
                u += x[i] * weights[i][j];
            out[j] = 1 / (1 + Math.exp(-u)); 
            }
        return out;
    }



}