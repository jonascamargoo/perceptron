import java.util.Random;

public class Perceptron {
    private double[][] weights;
    private double learningRate;
    private int numInputs;
    private int numOutputs;

    public Perceptron(int numInputs, int numOutputs, double learningRate) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
        this.weights = new double[numOutputs][numInputs + 1];
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs + 1; j++) {
                weights[i][j] = rand.nextDouble() - 0.5;
            }
        }
    }

    public double[] treinar(double[] inputs, double[] expectedOutput) {
        double[] x = new double[inputs.length + 1];
        x[0] = 1; // Bias term
        System.arraycopy(inputs, 0, x, 1, inputs.length);

        double[] outputs = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            double u = 0;
            for (int i = 0; i < numInputs + 1; i++) {
                u += x[i] * weights[j][i];
            }
            outputs[j] = 1 / (1 + Math.exp(-u));
        }

        for (int j = 0; j < numOutputs; j++) {
            double error = expectedOutput[j] - outputs[j];
            for (int i = 0; i < numInputs + 1; i++) {
                weights[j][i] += learningRate * error * x[i];
            }
        }
        return outputs;
    }

    public double[] executar(double[] inputs) {
        double[] x = new double[inputs.length + 1];
        x[0] = 1; // Bias term
        System.arraycopy(inputs, 0, x, 1, inputs.length);

        double[] outputs = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            double u = 0;
            for (int i = 0; i < numInputs + 1; i++) {
                u += x[i] * weights[j][i];
            }
            outputs[j] = 1 / (1 + Math.exp(-u));
        }
        return outputs;
    }
}