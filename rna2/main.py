# main.py

from perceptron import Perceptron
import data_processor
import training_utils
import plot_utils

def main():
    """
    Ponto de entrada principal para o fluxo de treinamento e avaliação do Perceptron.
    """
    # --- Hiperparâmetros e Configurações ---
    NUM_EPOCAS = 10000
    TAXA_APRENDIZADO = 0.1
    PASTA_RESULTADOS = "results" # Define o nome da pasta de saída

    # 1. Carregar e preparar os dados
    X_train, X_test, y_train, y_test = data_processor.load_and_prepare_data()

    num_entradas = X_train.shape[1]
    num_saidas = y_train.shape[1]

    # Instanciar o modelo
    perceptron_model = Perceptron(
        number_of_inputs=num_entradas, 
        number_of_outputs=num_saidas, 
        learning_rate=TAXA_APRENDIZADO
    )

    # 2. Treinar o modelo
    erros_aprox, erros_class = training_utils.train_perceptron(
        model=perceptron_model,
        X_train=X_train,
        y_train=y_train,
        num_epochs=NUM_EPOCAS
    )

    # 3. Avaliar o modelo treinado
    training_utils.evaluate_model(perceptron_model, X_test, y_test)

    # 4. Gerar e salvar o gráfico de erros na pasta definida
    plot_utils.save_error_plot(
        NUM_EPOCAS, 
        erros_aprox, 
        erros_class, 
        output_dir=PASTA_RESULTADOS
    )

    print("\nProcesso finalizado.")

if __name__ == "__main__":
    main()