package apresentacoes.apresentacao1;

import java.util.Random;

public class Perceptron extends RNA {
    private double[][] pesos;
    private double taxaDeAprendizado;
    private int numeroDeEntradas;
    private int numeroDeSaidas;

    public Perceptron(int numeroDeEntradas, int numeroDeSaidas, double taxaDeAprendizado) {
        this.numeroDeEntradas = numeroDeEntradas;
        this.numeroDeSaidas = numeroDeSaidas;
        this.taxaDeAprendizado = taxaDeAprendizado;
        this.pesos = new double[numeroDeSaidas][numeroDeEntradas + 1];
        inicializarPesos();
    }

    private void inicializarPesos() {
        Random aleatorio = new Random();
        for (int i = 0; i < numeroDeSaidas; i++) {
            for (int j = 0; j < numeroDeEntradas + 1; j++) {
                // Inicializa os pesos com valores aleatórios pequenos entre -0.5 e 0.5
                pesos[i][j] = aleatorio.nextDouble() - 0.5;
            }
        }
    }

    public double[] treinar(double[] entradas, double[] saidaEsperada) {
        // Adiciona o termo de bias às entradas
        double[] x = new double[entradas.length + 1];
        x[0] = 1; // Termo de bias
        System.arraycopy(entradas, 0, x, 1, entradas.length);

        // Calcula a saída da rede
        double[] saidas = new double[numeroDeSaidas];
        for (int j = 0; j < numeroDeSaidas; j++) {
            double u = 0;
            for (int i = 0; i < numeroDeEntradas + 1; i++) {
                u += x[i] * pesos[j][i];
            }
            // Aplica a função de ativação sigmoide
            saidas[j] = 1 / (1 + Math.exp(-u));
        }

        // Atualiza os pesos com base no erro
        for (int j = 0; j < numeroDeSaidas; j++) {
            double erro = saidaEsperada[j] - saidas[j];
            for (int i = 0; i < numeroDeEntradas + 1; i++) {
                pesos[j][i] += taxaDeAprendizado * erro * x[i];
            }
        }
        return saidas;
    }

    public double[] executar(double[] entradas) {
        // Adiciona o termo de bias às entradas
        double[] x = new double[entradas.length + 1];
        x[0] = 1; // Termo de bias
        System.arraycopy(entradas, 0, x, 1, entradas.length);

        // Calcula a saída da rede (sem treinar)
        double[] saidas = new double[numeroDeSaidas];
        for (int j = 0; j < numeroDeSaidas; j++) {
            double u = 0;
            for (int i = 0; i < numeroDeEntradas + 1; i++) {
                u += x[i] * pesos[j][i];
            }
            // Aplica a função de ativação sigmoide
            saidas[j] = 1 / (1 + Math.exp(-u));
        }
        return saidas;
    }
}