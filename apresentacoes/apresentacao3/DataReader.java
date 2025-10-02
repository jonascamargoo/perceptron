package apresentacoes.apresentacao3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    /**
     * Carrega, pré-processa e divide a base de dados de forma estratificada.
     * 1. Ordena as amostras por classe.
     * 2. Separa 75% de cada classe para treino (aleatoriamente).
     * 3. Usa os 25% restantes de cada classe para teste.
     */
    public static Dataset loadAndPrepareDataStratified(String filePath, double trainSize, int randomSeed) throws IOException {
        System.out.println("1. Carregando e preparando a base de dados de '" + filePath + "' de forma estratificada...");
        
        // Estrutura para agrupar as amostras por classe
        Map<Double, List<String[]>> recordsByClass = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Pula o cabeçalho
            while ((line = br.readLine()) != null) {
                if (line.contains("?") || line.contains(",,")) {
                    continue; // Pula linhas com dados ausentes
                }
                String[] values = line.split(",", -1);
                double targetClass = Double.parseDouble(values[values.length - 1]) > 0 ? 1.0 : 0.0;
                
                recordsByClass.putIfAbsent(targetClass, new ArrayList<>());
                recordsByClass.get(targetClass).add(values);
            }
        }
        
        List<double[]> X_train = new ArrayList<>();
        List<double[]> y_train = new ArrayList<>();
        List<double[]> X_test = new ArrayList<>();
        List<double[]> y_test = new ArrayList<>();
        
        Random random = new Random(randomSeed);

        // Processa cada classe separadamente
        for (Map.Entry<Double, List<String[]>> entry : recordsByClass.entrySet()) {
            Double currentClass = entry.getKey();
            List<String[]> classRecords = entry.getValue();
            
            Collections.shuffle(classRecords, random); // Embaralha as amostras da classe

            int trainSplitIndex = (int) (classRecords.size() * trainSize);

            // Adiciona 75% à base de treino
            for (int i = 0; i < trainSplitIndex; i++) {
                X_train.add(preprocessFeatures(classRecords.get(i)));
                y_train.add(new double[]{currentClass});
            }
            // Adiciona 25% à base de teste
            for (int i = trainSplitIndex; i < classRecords.size(); i++) {
                X_test.add(preprocessFeatures(classRecords.get(i)));
                y_test.add(new double[]{currentClass});
            }
        }
        
        // Normaliza os dados com base nos valores de treino
        normalizeData(X_train, X_test);

        System.out.println("Dados preparados com sucesso.");
        System.out.println(" -> Amostras de treino: " + X_train.size());
        System.out.println(" -> Amostras de teste:  " + X_test.size());
        
        return new Dataset(X_train, X_test, y_train, y_test);
    }

    private static double[] preprocessFeatures(String[] rawValues) {
        double[] features = new double[13];
        int featureIndex = 0;
        for (int j = 1; j < rawValues.length - 1; j++) {
            if (j == 3) continue; // Pula a coluna 'dataset'
            features[featureIndex++] = preprocessValue(j, rawValues[j]);
        }
        return features;
    }
    
    private static void normalizeData(List<double[]> trainData, List<double[]> testData) {
        if (trainData.isEmpty()) return;
        int numFeatures = trainData.get(0).length;
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            minValues[j] = Double.MAX_VALUE;
            maxValues[j] = Double.MIN_VALUE;
        }

        // Encontra min/max APENAS nos dados de treino
        for (double[] features : trainData) {
            for (int j = 0; j < numFeatures; j++) {
                if (features[j] < minValues[j]) minValues[j] = features[j];
                if (features[j] > maxValues[j]) maxValues[j] = features[j];
            }
        }

        // Normaliza os dados de treino
        for (double[] features : trainData) {
            for (int j = 0; j < numFeatures; j++) {
                if (maxValues[j] - minValues[j] != 0) {
                    features[j] = (features[j] - minValues[j]) / (maxValues[j] - minValues[j]);
                } else {
                    features[j] = 0;
                }
            }
        }
        
        // Normaliza os dados de teste USANDO os min/max do treino
        for (double[] features : testData) {
            for (int j = 0; j < numFeatures; j++) {
                 if (maxValues[j] - minValues[j] != 0) {
                    features[j] = (features[j] - minValues[j]) / (maxValues[j] - minValues[j]);
                } else {
                    features[j] = 0;
                }
            }
        }
    }

    private static double preprocessValue(int originalColumnIndex, String value) {
        // (O mesmo método preprocessValue da resposta anterior)
        switch (originalColumnIndex) {
            case 2: return value.equalsIgnoreCase("Male") ? 1.0 : 0.0;
            case 4: 
                if (value.equalsIgnoreCase("typical angina")) return 0.0;
                if (value.equalsIgnoreCase("atypical angina")) return 1.0;
                if (value.equalsIgnoreCase("non-anginal")) return 2.0;
                if (value.equalsIgnoreCase("asymptomatic")) return 3.0;
                break;
            case 7: return value.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 8:
                 if (value.equalsIgnoreCase("normal")) return 0.0;
                 if (value.equalsIgnoreCase("st-t abnormality")) return 1.0;
                 if (value.equalsIgnoreCase("lv hypertrophy")) return 2.0;
                 break;
            case 10: return value.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 12:
                if (value.equalsIgnoreCase("upsloping")) return 0.0;
                if (value.equalsIgnoreCase("flat")) return 1.0;
                if (value.equalsIgnoreCase("downsloping")) return 2.0;
                break;
            case 14:
                if (value.equalsIgnoreCase("normal")) return 0.0;
                if (value.equalsIgnoreCase("fixed defect")) return 1.0;
                if (value.equalsIgnoreCase("reversable defect")) return 2.0;
                break;
        }
        return Double.parseDouble(value.trim());
    }
}
