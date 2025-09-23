# training_utils.py

import numpy as np

def train_perceptron(model, X_train, y_train, num_epochs):
    """
    Executa o loop de treinamento do Perceptron e calcula os erros.

    Args:
        model: A instância do Perceptron a ser treinada.
        X_train: Dados de entrada de treinamento.
        y_train: Dados de saída de treinamento.
        num_epochs (int): Número de épocas para o treinamento.

    Retorna:
        (erros_aprox_plot, erros_class_plot): Tupla com as listas de erros normalizados.
    """
    print("\n2. Iniciando o treinamento do Perceptron...")
    
    num_amostras = X_train.shape[0]
    erros_aprox_plot = []
    erros_class_plot = []
    max_erro_aprox_epoca = 0.0

    for epoca in range(num_epochs):
        erro_aprox_da_epoca = 0.0
        erro_class_da_epoca = 0.0

        for i in range(num_amostras):
            entradas = X_train[i]
            saida_esperada = y_train[i]
            
            saida_calculada = model.train(entradas, saida_esperada)

            # Cálculo de erro de aproximação (Equações 1 e 2)
            erro_aprox_da_epoca += np.sum(np.abs(saida_esperada - saida_calculada))

            # Cálculo de erro de classificação (Equações 3, 4 e 5)
            saida_threshold = (saida_calculada >= 0.5).astype(int)
            if np.sum(np.abs(saida_esperada - saida_threshold)) > 0:
                erro_class_da_epoca += 1
        
        if (epoca + 1) % 1000 == 0: # Imprime a cada 1000 épocas para não poluir o console
            print(f"Época {epoca + 1}/{num_epochs} -> Erro Aproximação: {erro_aprox_da_epoca:.4f} | Erro Classificação: {int(erro_class_da_epoca)}")

        # Normalização dos erros para o gráfico (Equações 6 e 7)
        if erro_aprox_da_epoca > max_erro_aprox_epoca:
            max_erro_aprox_epoca = erro_aprox_da_epoca
        
        egraf_cl = erro_class_da_epoca / num_amostras
        egraf_ap = erro_aprox_da_epoca / max_erro_aprox_epoca if max_erro_aprox_epoca > 0 else 0.0
        
        erros_aprox_plot.append(egraf_ap)
        erros_class_plot.append(egraf_cl)
        
    print("Treinamento concluído.")
    return erros_aprox_plot, erros_class_plot

def evaluate_model(model, X_test, y_test):
    """
    Avalia a acurácia do modelo treinado no conjunto de teste.
    """
    print("\n3. Avaliando o modelo no conjunto de teste...")
    acertos = 0
    for i in range(len(X_test)):
        resultado = model.execute(X_test[i])
        predicao_final = (resultado >= 0.5).astype(int)
        if predicao_final == y_test[i]:
            acertos += 1
            
    acuracia = (acertos / len(X_test)) * 100
    print(f"-> Acurácia final no conjunto de teste: {acuracia:.2f}%")