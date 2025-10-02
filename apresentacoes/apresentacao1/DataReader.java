package apresentacoes.apresentacao1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

// Classe para encapsular os conjuntos de dados divididos
class Dataset {
    public final List<double[]> X_train, X_test;
    public final List<double[]> y_train, y_test;

    public Dataset(List<double[]> X_train, List<double[]> X_test, List<double[]> y_train, List<double[]> y_test) {
        this.X_train = X_train;
        this.X_test = X_test;
        this.y_train = y_train;
        this.y_test = y_test;
    }
}

public class DataReader {

    public static Dataset loadAndPrepareData(String filePath, double testSize, int randomSeed) throws IOException {
        System.out.println("1. Carregando e preparando a base de dados de '" + filePath + "'...");
        
        List<String[]> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Pula o cabeçalho do CSV
            while ((line = br.readLine()) != null) {
                // --- INÍCIO DA MODIFICAÇÃO ---
                // Pula a linha se ela contiver um '?' ou se, após o split, gerar um valor vazio.
                if (line.contains("?")) {
                    continue; // Pula a linha com dados ausentes marcados como '?'
                }

                String[] values = line.split(",", -1); // O -1 garante que strings vazias no final sejam contadas
                boolean hasEmptyValue = false;
                for (String value : values) {
                    if (value.trim().isEmpty()) {
                        hasEmptyValue = true;
                        break;
                    }
                }
                
                if (!hasEmptyValue) {
                    records.add(values);
                }
                // --- FIM DA MODIFICAÇÃO ---
            }
        }

        Collections.shuffle(records, new Random(randomSeed));

        // O CSV tem 16 colunas. Vamos usar 13 como features, ignorando 'id' e 'dataset'.
        int numFeatures = 13;
        double[][] features = new double[records.size()][numFeatures];
        double[][] targets = new double[records.size()][1];
        
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];
        for(int i = 0; i < numFeatures; i++) {
            minValues[i] = Double.MAX_VALUE;
            maxValues[i] = Double.MIN_VALUE;
        }

        // Primeira passagem: Ler, pré-processar e encontrar min/max
        for (int i = 0; i < records.size(); i++) {
            String[] values = records.get(i);
            int featureIndex = 0; // Índice para o nosso array de features
            
            for (int j = 1; j < values.length - 1; j++) { // Pula a coluna 'id' (j=0) e a 'num' (última)
                if (j == 3) continue; // Pula a coluna 'dataset' (j=3)
                
                double val = preprocessValue(j, values[j]);
                features[i][featureIndex] = val;

                if (val < minValues[featureIndex]) minValues[featureIndex] = val;
                if (val > maxValues[featureIndex]) maxValues[featureIndex] = val;
                
                featureIndex++;
            }
            // O alvo 'num' é convertido para binário (0 = sem doença, 1 = com doença)
            targets[i][0] = Double.parseDouble(values[values.length - 1]) > 0 ? 1.0 : 0.0;
        }

        // Segunda passagem: Normalizar os dados
        for (int i = 0; i < records.size(); i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (maxValues[j] - minValues[j] != 0) {
                     features[i][j] = (features[i][j] - minValues[j]) / (maxValues[j] - minValues[j]);
                } else {
                     features[i][j] = 0;
                }
            }
        }
        
        // Dividir em treino e teste
        int testSplitIndex = (int) (records.size() * (1 - testSize));
        
        List<double[]> X_train = new ArrayList<>();
        List<double[]> y_train = new ArrayList<>();
        List<double[]> X_test = new ArrayList<>();
        List<double[]> y_test = new ArrayList<>();

        for(int i = 0; i < testSplitIndex; i++) {
            X_train.add(features[i]);
            y_train.add(targets[i]);
        }
        for(int i = testSplitIndex; i < records.size(); i++) {
            X_test.add(features[i]);
            y_test.add(targets[i]);
        }

        System.out.println("Dados preparados com sucesso.");
        System.out.println(" -> Amostras de treino: " + X_train.size());
        System.out.println(" -> Amostras de teste:  " + X_test.size());
        
        return new Dataset(X_train, X_test, y_train, y_test);
    }

    /**
     * Converte valores categóricos (texto) para numéricos com base no índice ORIGINAL da coluna do CSV.
     */
    private static double preprocessValue(int originalColumnIndex, String value) {
        switch (originalColumnIndex) {
            case 2: // sex
                return value.equalsIgnoreCase("Male") ? 1.0 : 0.0;
            case 4: // cp (chest pain)
                if (value.equalsIgnoreCase("typical angina")) return 0.0;
                if (value.equalsIgnoreCase("atypical angina")) return 1.0;
                if (value.equalsIgnoreCase("non-anginal")) return 2.0;
                if (value.equalsIgnoreCase("asymptomatic")) return 3.0;
                break;
            case 7: // fbs (fasting blood sugar)
                return value.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 8: // restecg
                 if (value.equalsIgnoreCase("normal")) return 0.0;
                 if (value.equalsIgnoreCase("st-t abnormality")) return 1.0; 
                 if (value.equalsIgnoreCase("lv hypertrophy")) return 2.0;
                 break;
            case 10: // exang (exercise angina)
                return value.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 12: // slope
                if (value.equalsIgnoreCase("upsloping")) return 0.0;
                if (value.equalsIgnoreCase("flat")) return 1.0;
                if (value.equalsIgnoreCase("downsloping")) return 2.0;
                break;
            case 14: // thal
                if (value.equalsIgnoreCase("normal")) return 0.0;
                if (value.equalsIgnoreCase("fixed defect")) return 1.0;
                if (value.equalsIgnoreCase("reversable defect")) return 2.0;
                break;
        }
        // Se não for uma coluna categórica, tenta converter para double
        return Double.parseDouble(value.trim());
    }
}