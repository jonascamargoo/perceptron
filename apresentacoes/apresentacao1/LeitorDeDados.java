package apresentacoes.apresentacao1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

class ConjuntoDeDados {
    public final List<double[]> X_treino, X_teste;
    public final List<double[]> y_treino, y_teste;

    public ConjuntoDeDados(List<double[]> X_treino, List<double[]> X_teste, List<double[]> y_treino, List<double[]> y_teste) {
        this.X_treino = X_treino;
        this.X_teste = X_teste;
        this.y_treino = y_treino;
        this.y_teste = y_teste;
    }
}

public class LeitorDeDados {

    public static ConjuntoDeDados carregarEPrepararDados(String caminhoArquivo, double proporcaoTeste, int sementeAleatoria) throws IOException {
        System.out.println("1. Carregando e preparando a base de dados de '" + caminhoArquivo + "'...");
        
        List<String[]> registros = new ArrayList<>();
        try (BufferedReader leitor = new BufferedReader(new FileReader(caminhoArquivo))) {
            String linha;
            leitor.readLine();
            while ((linha = leitor.readLine()) != null) {
                if (linha.contains("?")) {
                    continue; // Pula a linha com dados ausentes marcados como '?'
                }

                String[] valores = linha.split(",", -1); // O -1 garante que strings vazias no final sejam contadas
                boolean temValorVazio = false;
                for (String valor : valores) {
                    if (valor.trim().isEmpty()) {
                        temValorVazio = true;
                        break;
                    }
                }
                
                if (!temValorVazio) {
                    registros.add(valores);
                }
            }
        }

        Collections.shuffle(registros, new Random(sementeAleatoria));

        // O CSV tem 16 colunas. Vamos usar 13 como características, ignorando 'id' e 'dataset
        int numCaracteristicas = 13;
        double[][] caracteristicas = new double[registros.size()][numCaracteristicas];
        double[][] alvos = new double[registros.size()][1];
        
        double[] valoresMinimos = new double[numCaracteristicas];
        double[] valoresMaximos = new double[numCaracteristicas];
        for(int i = 0; i < numCaracteristicas; i++) {
            valoresMinimos[i] = Double.MAX_VALUE;
            valoresMaximos[i] = Double.MIN_VALUE;
        }

        // Primeira passagem: Ler, pré-processar e encontrar min/max
        for (int i = 0; i < registros.size(); i++) {
            String[] valores = registros.get(i);
            int indiceCaracteristica = 0; // Índice para o nosso array de características
            
            for (int j = 1; j < valores.length - 1; j++) { // Pula a coluna 'id' (j=0) e a 'num' (última)
                if (j == 3) continue; // Pula a coluna 'dataset' (j=3)
                
                double valorProcessado = preprocessarValor(j, valores[j]);
                caracteristicas[i][indiceCaracteristica] = valorProcessado;

                if (valorProcessado < valoresMinimos[indiceCaracteristica]) valoresMinimos[indiceCaracteristica] = valorProcessado;
                if (valorProcessado > valoresMaximos[indiceCaracteristica]) valoresMaximos[indiceCaracteristica] = valorProcessado;
                
                indiceCaracteristica++;
            }
            // O alvo 'num' é convertido para binário (0 = sem doença, 1 = com doença)
            alvos[i][0] = Double.parseDouble(valores[valores.length - 1]) > 0 ? 1.0 : 0.0;
        }

        // Segunda passagem: Normalizar os dados
        for (int i = 0; i < registros.size(); i++) {
            for (int j = 0; j < numCaracteristicas; j++) {
                if (valoresMaximos[j] - valoresMinimos[j] != 0) {
                     caracteristicas[i][j] = (caracteristicas[i][j] - valoresMinimos[j]) / (valoresMaximos[j] - valoresMinimos[j]);
                } else {
                     caracteristicas[i][j] = 0;
                }
            }
        }
        
        // Dividir em treino e teste
        int indiceDivisaoTeste = (int) (registros.size() * (1 - proporcaoTeste));
        
        List<double[]> X_treino = new ArrayList<>();
        List<double[]> y_treino = new ArrayList<>();
        List<double[]> X_teste = new ArrayList<>();
        List<double[]> y_teste = new ArrayList<>();

        for(int i = 0; i < indiceDivisaoTeste; i++) {
            X_treino.add(caracteristicas[i]);
            y_treino.add(alvos[i]);
        }
        for(int i = indiceDivisaoTeste; i < registros.size(); i++) {
            X_teste.add(caracteristicas[i]);
            y_teste.add(alvos[i]);
        }

        System.out.println("Dados preparados com sucesso.");
        System.out.println(" -> Amostras de treino: " + X_treino.size());
        System.out.println(" -> Amostras de teste:  " + X_teste.size());
        
        return new ConjuntoDeDados(X_treino, X_teste, y_treino, y_teste);
    }

    /**
     * Converte valores categóricos (texto) para numéricos com base no índice ORIGINAL da coluna do CSV.
     */
    private static double preprocessarValor(int indiceColunaOriginal, String valor) {
        switch (indiceColunaOriginal) {
            case 2: // sex
                return valor.equalsIgnoreCase("Male") ? 1.0 : 0.0;
            case 4: // cp (chest pain / dor no peito)
                if (valor.equalsIgnoreCase("typical angina")) return 0.0;
                if (valor.equalsIgnoreCase("atypical angina")) return 1.0;
                if (valor.equalsIgnoreCase("non-anginal")) return 2.0;
                if (valor.equalsIgnoreCase("asymptomatic")) return 3.0;
                break;
            case 7: // fbs (fasting blood sugar / açúcar no sangue em jejum)
                return valor.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 8: // restecg (eletrocardiograma em repouso)
                 if (valor.equalsIgnoreCase("normal")) return 0.0;
                 if (valor.equalsIgnoreCase("st-t abnormality")) return 1.0; 
                 if (valor.equalsIgnoreCase("lv hypertrophy")) return 2.0;
                 break;
            case 10: // exang (angina induzida por exercício)
                return valor.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 12: // slope (inclinação do segmento ST)
                if (valor.equalsIgnoreCase("upsloping")) return 0.0;
                if (valor.equalsIgnoreCase("flat")) return 1.0;
                if (valor.equalsIgnoreCase("downsloping")) return 2.0;
                break;
            case 14: // thal (talassemia)
                if (valor.equalsIgnoreCase("normal")) return 0.0;
                if (valor.equalsIgnoreCase("fixed defect")) return 1.0;
                if (valor.equalsIgnoreCase("reversable defect")) return 2.0;
                break;
        }
        // Se não for uma coluna categórica, tenta converter para double
        return Double.parseDouble(valor.trim());
    }
}