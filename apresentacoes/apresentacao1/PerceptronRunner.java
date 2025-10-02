package apresentacoes.apresentacao1;

import java.io.IOException;

public class PerceptronRunner {

    public static void main(String[] args) {
        // --- Hiperparâmetros e Configurações ---
        int NUMERO_DE_EPOCAS = 10000;
        double TAXA_DE_APRENDIZADO = 0.1;
        String NOME_ARQUIVO_DADOS = "heart_disease_uci.csv"; // O arquivo deve estar na raiz do projeto

        try {
            // 1. Carregar e preparar os dados usando as classes em português
            ConjuntoDeDados conjuntoDeDados = LeitorDeDados.carregarEPrepararDados(NOME_ARQUIVO_DADOS, 0.2, 42);
            
            int numEntradas = conjuntoDeDados.X_treino.get(0).length;
            int numSaidas = conjuntoDeDados.y_treino.get(0).length;

            // Instanciar o modelo
            Perceptron perceptron = new Perceptron(numEntradas, numSaidas, TAXA_DE_APRENDIZADO);

            // 2. Treinar o modelo
            treinarModelo(perceptron, conjuntoDeDados, NUMERO_DE_EPOCAS);

            // 3. Avaliar o modelo treinado
            avaliarModelo(perceptron, conjuntoDeDados);

        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo de dados: " + NOME_ARQUIVO_DADOS);
            System.err.println("Certifique-se de que o arquivo '" + NOME_ARQUIVO_DADOS + "' está na pasta raiz do seu projeto.");
            e.printStackTrace();
        }
         System.out.println("\nProcesso finalizado.");
    }

    public static void treinarModelo(Perceptron modelo, ConjuntoDeDados dados, int numEpocas) {
        System.out.println("\n2. Iniciando o treinamento do Perceptron...");
        
        for (int epoca = 0; epoca < numEpocas; epoca++) {
            double erroAproxDaEpoca = 0.0;
            int erroClassDaEpoca = 0;

            for (int i = 0; i < dados.X_treino.size(); i++) {
                double[] entradas = dados.X_treino.get(i);
                double[] saidaEsperada = dados.y_treino.get(i);
                
                double[] saidaCalculada = modelo.treinar(entradas, saidaEsperada);

                // Cálculo de erro de aproximação
                erroAproxDaEpoca += Math.abs(saidaEsperada[0] - saidaCalculada[0]);
                
                // Cálculo de erro de classificação
                double saidaComLimiar = (saidaCalculada[0] >= 0.5) ? 1.0 : 0.0;
                if (saidaComLimiar != saidaEsperada[0]) {
                    erroClassDaEpoca++;
                }
            }
            
            if ((epoca + 1) % 1000 == 0) {
                 System.out.printf("Época %5d/%d -> Erro Aproximação: %8.4f | Erro Classificação: %d\n", 
                                   epoca + 1, numEpocas, erroAproxDaEpoca, erroClassDaEpoca);
            }
        }
        System.out.println("Treinamento concluído.");
    }
    
    public static void avaliarModelo(Perceptron modelo, ConjuntoDeDados dados) {
        System.out.println("\n3. Avaliando o modelo no conjunto de teste...");
        int acertos = 0;
        for (int i = 0; i < dados.X_teste.size(); i++) {
            double[] entradas = dados.X_teste.get(i);
            double[] resultado = modelo.executar(entradas);
            
            // Converte a saída contínua para uma classe (0 ou 1)
            double predicaoFinal = (resultado[0] >= 0.5) ? 1.0 : 0.0;
            
            if (predicaoFinal == dados.y_teste.get(i)[0]) {
                acertos++;
            }
        }
        
        double acuracia = ((double) acertos / dados.X_teste.size()) * 100.0;
        System.out.printf("-> Acurácia final no conjunto de teste: %.2f%%\n", acuracia);
    }
}