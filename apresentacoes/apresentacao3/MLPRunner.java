package apresentacoes.apresentacao3; // Ajuste o nome do pacote se necessário

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class MLPRunner {

    public static void main(String[] args) {
        // --- Hiperparâmetros e Configurações ---
        int NUMERO_DE_EPOCAS = 10000;
        double TAXA_DE_APRENDIZADO = 0.1;
        int NEURONIOS_CAMADA_OCULTA = 10;
        String NOME_ARQUIVO_DADOS = "heart_disease_uci.csv";
        String NOME_ARQUIVO_LOG = "log_de_treinamento.csv"; // Arquivo de saída para o gráfico

        try {
            // 1. Carregar e preparar os dados usando a classe e o método em português
            ConjuntoDeDados conjuntoDeDados = LeitorDeDados.carregarEPrepararDadosEstratificados(NOME_ARQUIVO_DADOS, 0.75, 42);
            
            int numEntradas = conjuntoDeDados.X_treino.get(0).length;
            int numSaidas = conjuntoDeDados.y_treino.get(0).length;

            // Instanciar o modelo MLP
            MLP mlp = new MLP(numEntradas, NEURONIOS_CAMADA_OCULTA, numSaidas, TAXA_DE_APRENDIZADO);

            System.out.println("\n2. Iniciando o treinamento e avaliação do MLP...");
            
            // Abre o arquivo para salvar o log do treinamento
            try (PrintWriter escritorLog = new PrintWriter(new FileWriter(NOME_ARQUIVO_LOG))) {
                
                // Escreve o cabeçalho no CSV e no console
                String cabecalho = String.format("%-7s, %-18s, %-18s, %-18s, %-18s", 
                    "Epoca", "ErroAproxTreino", "ErroClassTreino", "ErroAproxTeste", "ErroClassTeste");
                escritorLog.println(cabecalho.replace(" ", "")); // Salva sem espaços no CSV
                System.out.println(cabecalho.replace(",", " |"));

                // --- LOOP DE TREINAMENTO E TESTE ---
                for (int epoca = 0; epoca < NUMERO_DE_EPOCAS; epoca++) {
                    
                    // --- FASE DE TREINO ---
                    double erroAproxTreino = 0.0;
                    int erroClassTreino = 0;
                    for (int i = 0; i < conjuntoDeDados.X_treino.size(); i++) {
                        double[] entradas = conjuntoDeDados.X_treino.get(i);
                        double[] saidaEsperada = conjuntoDeDados.y_treino.get(i);
                        double[] saidaCalculada = mlp.treinar(entradas, saidaEsperada);
                        
                        erroAproxTreino += Math.abs(saidaEsperada[0] - saidaCalculada[0]);
                        if ((saidaCalculada[0] >= 0.5 ? 1.0 : 0.0) != saidaEsperada[0]) {
                            erroClassTreino++;
                        }
                    }

                    // --- FASE DE TESTE ---
                    double erroAproxTeste = 0.0;
                    int erroClassTeste = 0;
                    for (int i = 0; i < conjuntoDeDados.X_teste.size(); i++) {
                        double[] entradas = conjuntoDeDados.X_teste.get(i);
                        double[] saidaEsperada = conjuntoDeDados.y_teste.get(i);
                        double[] saidaCalculada = mlp.executar(entradas); // Usa executar(), não treinar()!
                        
                        erroAproxTeste += Math.abs(saidaEsperada[0] - saidaCalculada[0]);
                         if ((saidaCalculada[0] >= 0.5 ? 1.0 : 0.0) != saidaEsperada[0]) {
                            erroClassTeste++;
                        }
                    }

                    // Salva os dados da época no arquivo CSV
                    String linhaLog = String.format("%d,%.4f,%d,%.4f,%d", 
                        epoca + 1, erroAproxTreino, erroClassTreino, erroAproxTeste, erroClassTeste);
                    escritorLog.println(linhaLog);

                    // Imprime o progresso no console a cada 100 épocas
                    if ((epoca + 1) % 100 == 0) { 
                        System.out.printf("%-7d | %-18.4f | %-18d | %-18.4f | %-18d\n", 
                            epoca + 1, erroAproxTreino, erroClassTreino, erroAproxTeste, erroClassTeste);
                    }
                }
            } 

        } catch (IOException e) {
            System.err.println("Ocorreu um erro: " + e.getMessage());
            e.printStackTrace();
        }
         System.out.println("\nLog de treinamento salvo em '" + NOME_ARQUIVO_LOG + "'.");
         System.out.println("Processo finalizado.");
    }
}