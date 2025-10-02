package apresentacoes.apresentacao2;

import java.util.Arrays;

public class MLPRunner {

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

    // XOR - O problema que o Perceptron não resolve!
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
        // Para o MLP, o teste mais importante é o XOR.
        // Vamos definir uma arquitetura com:
        // - 2 neurônios de entrada
        // - 4 neurônios na camada oculta (um bom ponto de partida)
        // - 1 neurônio de saída
        treinarETestar("Porta Lógica XOR", DADOS_PORTA_XOR, 2, 4, 1);
        
        // Você também pode testar os outros problemas
        // treinarETestar("Porta Lógica AND", DADOS_PORTA_AND, 2, 4, 1);
        // treinarETestar("Porta Lógica OR", DADOS_PORTA_OR, 2, 4, 1);
        // treinarETestar("Robô", DADOS_ROBO, 3, 5, 2); // Aumentando um pouco a camada oculta para o robô
    }

    public static void treinarETestar(String nomeDoTeste, double[][][] dados, int numEntradas, int numOcultos, int numSaidas) {
        System.out.println("### Treinando e Testando MLP: " + nomeDoTeste + " ###");
        MLP mlp = new MLP(numEntradas, numOcultos, numSaidas, 0.3);

        int epocas = 10000;
        for (int epocaAtual = 0; epocaAtual < epocas; epocaAtual++) {
            double erroDaEpoca = 0;
            for (double[][] amostra : dados) {
                double[] entradas = amostra[0];
                double[] saidaEsperada = amostra[1];
                double[] saidaCalculada = mlp.treinar(entradas, saidaEsperada);
                
                // Soma o erro absoluto da amostra atual
                for (int i = 0; i < saidaEsperada.length; i++) {
                    erroDaEpoca += Math.abs(saidaEsperada[i] - saidaCalculada[i]);
                }
            }
            
            // Imprime o erro a cada 1000 épocas para não poluir o console
            if ((epocaAtual + 1) % 1000 == 0) {
                 System.out.println((epocaAtual + 1) + "ª época, erro de aproximação: " + erroDaEpoca);
            }
        }

        // Testa a rede após o treinamento
        System.out.println("\nResultados após o treinamento para " + nomeDoTeste + ":");
        for (double[][] amostra : dados) {
            double[] entradas = amostra[0];
            double[] resultado = mlp.executar(entradas);
            // Arredonda a saída para ficar mais legível (ex: 0.99... vira 1.0, 0.01... vira 0.0)
            double[] resultadoArredondado = new double[resultado.length];
            for (int i=0; i < resultado.length; i++) {
                resultadoArredondado[i] = Math.round(resultado[i]);
            }

            System.out.println("Entrada: " + Arrays.toString(entradas) + 
                               " Saída: " + Arrays.toString(resultadoArredondado) + 
                               " Esperado: " + Arrays.toString(amostra[1]));
        }
        System.out.println("########################################\n");
    }
}