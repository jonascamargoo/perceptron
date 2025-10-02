package apresentacoes.apresentacao3;

import java.io.IOException;
import java.util.List;

public class MLPRunner {

    public static void main(String[] args) {
        // --- Hiperparâmetros e Configurações ---
        int NUM_EPOCAS = 10000;
        double TAXA_APRENDIZADO = 0.1;
        int NEURONIOS_CAMADA_OCULTA = 10; // Um bom ponto de partida
        String NOME_ARQUIVO_DADOS = "heart_disease_uci.csv";

        try {
            // 1. Carregar e preparar os dados
            Dataset dataset = DataReader.loadAndPrepareDataStratified(NOME_ARQUIVO_DADOS, 0.75, 42);
            
            int numEntradas = dataset.X_train.get(0).length;
            int numSaidas = dataset.y_train.get(0).length;

            // Instanciar o modelo MLP
            MLP mlp = new MLP(numEntradas, NEURONIOS_CAMADA_OCULTA, numSaidas, TAXA_APRENDIZADO);

            System.out.println("\n2. Iniciando o treinamento e avaliação do MLP...");
            System.out.printf("%-7s | %-18s | %-18s | %-18s | %-18s\n", 
                "Época", "Erro Aprox. Treino", "Erro Class. Treino", "Erro Aprox. Teste", "Erro Class. Teste");
            
            // --- NOVO LOOP DE TREINAMENTO E TESTE ---
            for (int epoca = 0; epoca < NUM_EPOCAS; epoca++) {
                
                // --- FASE DE TREINO ---
                double erroAproxTreino = 0.0;
                int erroClassTreino = 0;
                for (int i = 0; i < dataset.X_train.size(); i++) {
                    double[] entradas = dataset.X_train.get(i);
                    double[] saidaEsperada = dataset.y_train.get(i);
                    double[] saidaCalculada = mlp.treinar(entradas, saidaEsperada);
                    
                    erroAproxTreino += Math.abs(saidaEsperada[0] - saidaCalculada[0]);
                    if ((saidaCalculada[0] >= 0.5 ? 1.0 : 0.0) != saidaEsperada[0]) {
                        erroClassTreino++;
                    }
                }

                // --- FASE DE TESTE ---
                double erroAproxTeste = 0.0;
                int erroClassTeste = 0;
                for (int i = 0; i < dataset.X_test.size(); i++) {
                    double[] entradas = dataset.X_test.get(i);
                    double[] saidaEsperada = dataset.y_test.get(i);
                    double[] saidaCalculada = mlp.executar(entradas); // Usa executar(), não treinar()!
                    
                    erroAproxTeste += Math.abs(saidaEsperada[0] - saidaCalculada[0]);
                     if ((saidaCalculada[0] >= 0.5 ? 1.0 : 0.0) != saidaEsperada[0]) {
                        erroClassTeste++;
                    }
                }

                if ((epoca + 1) % 100 == 0) { // Imprime a cada 100 épocas
                    System.out.printf("%-7d | %-18.4f | %-18d | %-18.4f | %-18d\n", 
                        epoca + 1, erroAproxTreino, erroClassTreino, erroAproxTeste, erroClassTeste);
                }
            }

        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo de dados: " + NOME_ARQUIVO_DADOS);
            e.printStackTrace();
        }
         System.out.println("\nProcesso finalizado.");
    }
}