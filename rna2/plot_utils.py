import matplotlib.pyplot as plt
import os

def save_error_plot(epoch_count, errors_aprox, errors_class, output_dir="results", filename="grafico_erros.png"):
    """
    Gera e salva um gráfico da evolução dos erros de treinamento em um diretório específico.

    Args:
        epoch_count (int): O número total de épocas.
        errors_aprox (list): Lista com os erros de aproximação normalizados.
        errors_class (list): Lista com os erros de classificação normalizados.
        output_dir (str): O nome da pasta onde o gráfico será salvo.
        filename (str): Nome do arquivo para salvar o gráfico.
    """
    if not os.path.exists(output_dir):
        print(f"Criando o diretório: '{output_dir}'")
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    print(f"\n4. Salvando o gráfico de erros como '{filepath}'...")
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epoch_count + 1), errors_aprox, label='Erro de Aproximação Normalizado (Eq. 7)')
    plt.plot(range(1, epoch_count + 1), errors_class, label='Erro de Classificação Normalizado (Eq. 6)')
    plt.title('Evolução dos Erros Durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Erro Normalizado')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filepath)
    plt.close()
    print("Gráfico salvo com sucesso.")