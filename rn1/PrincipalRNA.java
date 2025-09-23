public class PrincipalRNA {

    // AND
    public static final double[][][] DADOS_PORTA_AND = {
            {{0, 0}, {0}},
            {{0, 1}, {0}},
            {{1, 0}, {0}},
            {{1, 1}, {1}}
    };

    // OR
    public static final double[][][] DADOS_PORTA_OR = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {1}}
    };

    // XOR
    public static final double[][][] DADOS_PORTA_XOR = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
    };

    // Robô
    public static final double[][][] DADOS_ROBO = {
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
        treinarETestar("Porta Lógica AND", DADOS_PORTA_AND, 2, 1);
        // treinarETestar("Porta Lógica OR", DADOS_PORTA_OR, 2, 1);
        // treinarETestar("Porta Lógica XOR", DADOS_PORTA_XOR, 2, 1);
        // treinarETestar("Robô", DADOS_ROBO, 3, 2);
    }

    public static void treinarETestar(String nomeDoTeste, double[][][] dados, int numEntradas, int numSaidas) {
        System.out.println("### Treinando e Testando: " + nomeDoTeste + " ###");
        // Cria um novo Perceptron com 30% de taxa de aprendizado.
        Perceptron p = new Perceptron(numEntradas, numSaidas, 0.3);

        // Treina a rede por 10.000 épocas
        for (int epoca = 0; epoca < 10000; epoca++) {
            double erroDaEpoca = 0;
            for (double[][] amostra : dados) {
                double[] entradas = amostra[0];
                double[] saidaEsperada = amostra[1];
                double[] saidaCalculada = p.treinar(entradas, saidaEsperada);
                // Soma o erro absoluto da amostra atual
                for (int i = 0; i < saidaEsperada.length; i++) {
                    erroDaEpoca += Math.abs(saidaEsperada[i] - saidaCalculada[i]);
                }
            }
            
            System.out.println((epoca + 1) + "ª época, erro de aproximação: " + erroDaEpoca);
        }

        // // Testa a rede após o treinamento
        // System.out.println("\nResultados após o treinamento para " + nomeDoTeste + ":");
        // for (double[][] amostra : dados) {
        //     double[] entradas = amostra[0];
        //     double[] resultado = p.executar(entradas);
        //     System.out.println("Entrada: " + Arrays.toString(entradas) + " Saída: " + Arrays.toString(resultado) + " Esperado: " + Arrays.toString(amostra[1]));
        // }
        // System.out.println("########################################\n");
    }
}