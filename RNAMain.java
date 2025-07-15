import java.util.Arrays;

public class RNAMain {

    public static final double[][][] AND_GATE_DATA = {
            {{0, 0}, {0}},
            {{0, 1}, {0}},
            {{1, 0}, {0}},
            {{1, 1}, {1}}
    };

    public static final double[][][] OR_GATE_DATA = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {1}}
    };

    public static final double[][][] XOR_GATE_DATA = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
    };

    public static final double[][][] ROBOT_DATA = {
        {{0, 0, 0}, {1, 1}},
        {{0, 0, 1}, {0, 1}},
        {{0, 1, 0}, {1, 0}},
        {{0, 1, 1}, {0, 1}},
        {{1, 0, 0}, {1, 0}},
        {{1, 0, 1}, {1, 0}},
        {{1, 1, 0}, {1, 0}},
        {{1, 1, 1}, {1, 0}}
    };


    public static void main(String[] args) {
        trainAndTest("Porta Lógica AND", AND_GATE_DATA, 2, 1);
        trainAndTest("Porta Lógica OR", OR_GATE_DATA, 2, 1);
        trainAndTest("Porta Lógica XOR", XOR_GATE_DATA, 2, 1);
        trainAndTest("Robô", ROBOT_DATA, 3, 2);
    }

    public static void trainAndTest(String testName, double[][][] data, int numInputs, int numOutputs) {
        System.out.println("### Treinando e Testando: " + testName + " ###");
        Perceptron p = new Perceptron(numInputs, numOutputs, 0.3);

        for (int epoch = 0; epoch < 10000; epoch++) {
            double epochError = 0;
            for (double[][] sample : data) {
                double[] inputs = sample[0];
                double[] expectedOutput = sample[1];
                double[] output = p.treinar(inputs, expectedOutput);
                for (int i = 0; i < expectedOutput.length; i++) {
                    epochError += Math.abs(expectedOutput[i] - output[i]);
                }
            }
            System.out.println((epoch + 1) + "ª da época erro de aproximação da época: " + epochError);
        }

        System.out.println("\nResultados após o treinamento para " + testName + ":");
        for (double[][] sample : data) {
            double[] inputs = sample[0];
            double[] result = p.executar(inputs);
            System.out.println("Entrada: " + Arrays.toString(inputs) + " Saída: " + Arrays.toString(result) + " Esperado: " + Arrays.toString(sample[1]));
        }
        System.out.println("########################################\n");
    }
}