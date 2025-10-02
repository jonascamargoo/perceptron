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
    /**
     * Carrega, pré-processa e divide a base de dados de forma estratificada.
     * 1. Agrupa as amostras por classe.
     * 2. Separa uma porcentagem de cada classe para treino (aleatoriamente).
     * 3. Usa o restante de cada classe para teste.
     */
    public static ConjuntoDeDados carregarEPrepararDadosEstratificados(String caminhoArquivo, double proporcaoTreino, int sementeAleatoria) throws IOException {
        System.out.println("1. Carregando e preparando a base de dados de '" + caminhoArquivo + "' de forma estratificada...");
        
        // Estrutura para agrupar as amostras por classe
        Map<Double, List<String[]>> registrosPorClasse = new HashMap<>();

        try (BufferedReader leitor = new BufferedReader(new FileReader(caminhoArquivo))) {
            String linha;
            leitor.readLine(); // Pula o cabeçalho do CSV
            while ((linha = leitor.readLine()) != null) {
                if (linha.contains("?") || linha.contains(",,")) {
                    continue; // Pula linhas com dados ausentes
                }
                String[] valores = linha.split(",", -1);
                double classeAlvo = Double.parseDouble(valores[valores.length - 1]) > 0 ? 1.0 : 0.0;
                
                registrosPorClasse.putIfAbsent(classeAlvo, new ArrayList<>());
                registrosPorClasse.get(classeAlvo).add(valores);
            }
        }
        
        List<double[]> X_treino = new ArrayList<>();
        List<double[]> y_treino = new ArrayList<>();
        List<double[]> X_teste = new ArrayList<>();
        List<double[]> y_teste = new ArrayList<>();
        
        Random aleatorio = new Random(sementeAleatoria);

        // Processa cada classe separadamente
        for (Map.Entry<Double, List<String[]>> item : registrosPorClasse.entrySet()) {
            Double classeAtual = item.getKey();
            List<String[]> registrosDaClasse = item.getValue();
            
            Collections.shuffle(registrosDaClasse, aleatorio); // Embaralha as amostras da classe

            int indiceDivisaoTreino = (int) (registrosDaClasse.size() * proporcaoTreino);

            // Adiciona a proporção definida para a base de treino
            for (int i = 0; i < indiceDivisaoTreino; i++) {
                X_treino.add(preprocessarCaracteristicas(registrosDaClasse.get(i)));
                y_treino.add(new double[]{classeAtual});
            }
            // Adiciona o restante à base de teste
            for (int i = indiceDivisaoTreino; i < registrosDaClasse.size(); i++) {
                X_teste.add(preprocessarCaracteristicas(registrosDaClasse.get(i)));
                y_teste.add(new double[]{classeAtual});
            }
        }
        
        // Normaliza os dados com base nos valores encontrados no conjunto de treino
        normalizarDados(X_treino, X_teste);

        System.out.println("Dados preparados com sucesso.");
        System.out.println(" -> Amostras de treino: " + X_treino.size());
        System.out.println(" -> Amostras de teste:  " + X_teste.size());
        
        return new ConjuntoDeDados(X_treino, X_teste, y_treino, y_teste);
    }

    private static double[] preprocessarCaracteristicas(String[] valoresBrutos) {
        double[] caracteristicas = new double[13];
        int indiceCaracteristica = 0;
        for (int j = 1; j < valoresBrutos.length - 1; j++) {
            if (j == 3) continue; // Pula a coluna 'dataset'
            caracteristicas[indiceCaracteristica++] = preprocessarValor(j, valoresBrutos[j]);
        }
        return caracteristicas;
    }
    
    private static void normalizarDados(List<double[]> dadosTreino, List<double[]> dadosTeste) {
        if (dadosTreino.isEmpty()) return;
        int numCaracteristicas = dadosTreino.get(0).length;
        double[] valoresMinimos = new double[numCaracteristicas];
        double[] valoresMaximos = new double[numCaracteristicas];

        for (int j = 0; j < numCaracteristicas; j++) {
            valoresMinimos[j] = Double.MAX_VALUE;
            valoresMaximos[j] = Double.MIN_VALUE;
        }

        // Encontra os valores mínimo e máximo APENAS nos dados de treino
        for (double[] caracteristicas : dadosTreino) {
            for (int j = 0; j < numCaracteristicas; j++) {
                if (caracteristicas[j] < valoresMinimos[j]) valoresMinimos[j] = caracteristicas[j];
                if (caracteristicas[j] > valoresMaximos[j]) valoresMaximos[j] = caracteristicas[j];
            }
        }

        // Normaliza os dados de treino
        for (double[] caracteristicas : dadosTreino) {
            for (int j = 0; j < numCaracteristicas; j++) {
                if (valoresMaximos[j] - valoresMinimos[j] != 0) {
                    caracteristicas[j] = (caracteristicas[j] - valoresMinimos[j]) / (valoresMaximos[j] - valoresMinimos[j]);
                } else {
                    caracteristicas[j] = 0;
                }
            }
        }
        
        // Normaliza os dados de teste USANDO os valores min/max do treino
        for (double[] caracteristicas : dadosTeste) {
            for (int j = 0; j < numCaracteristicas; j++) {
                 if (valoresMaximos[j] - valoresMinimos[j] != 0) {
                    caracteristicas[j] = (caracteristicas[j] - valoresMinimos[j]) / (valoresMaximos[j] - valoresMinimos[j]);
                } else {
                    caracteristicas[j] = 0;
                }
            }
        }
    }

    private static double preprocessarValor(int indiceColunaOriginal, String valor) {
        // Converte valores categóricos (texto) para numéricos
        switch (indiceColunaOriginal) {
            case 2: return valor.equalsIgnoreCase("Male") ? 1.0 : 0.0;
            case 4: 
                if (valor.equalsIgnoreCase("typical angina")) return 0.0;
                if (valor.equalsIgnoreCase("atypical angina")) return 1.0;
                if (valor.equalsIgnoreCase("non-anginal")) return 2.0;
                if (valor.equalsIgnoreCase("asymptomatic")) return 3.0;
                break;
            case 7: return valor.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 8:
                 if (valor.equalsIgnoreCase("normal")) return 0.0;
                 if (valor.equalsIgnoreCase("st-t abnormality")) return 1.0;
                 if (valor.equalsIgnoreCase("lv hypertrophy")) return 2.0;
                 break;
            case 10: return valor.equalsIgnoreCase("TRUE") ? 1.0 : 0.0;
            case 12:
                if (valor.equalsIgnoreCase("upsloping")) return 0.0;
                if (valor.equalsIgnoreCase("flat")) return 1.0;
                if (valor.equalsIgnoreCase("downsloping")) return 2.0;
                break;
            case 14:
                if (valor.equalsIgnoreCase("normal")) return 0.0;
                if (valor.equalsIgnoreCase("fixed defect")) return 1.0;
                if (valor.equalsIgnoreCase("reversable defect")) return 2.0;
                break;
        }
        return Double.parseDouble(valor.trim());
    }
}
