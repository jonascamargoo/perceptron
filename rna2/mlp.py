import numpy as np

class MLP:
    """
    Uma tradução da classe MLP de Java para Python usando NumPy.
    """
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate=0.3):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        # Inicializa os pesos com o bias incluído (+1 na dimensão de entrada)
        # wh: pesos da camada de entrada para a oculta
        # wo: pesos da camada oculta para a de saída
        self.wh = np.random.uniform(-0.3, 0.3, size=(num_inputs + 1, num_hidden))
        self.wo = np.random.uniform(-0.3, 0.3, size=(num_hidden + 1, num_outputs))

    def _sigmoid(self, x):
        """Função de ativação Sigmoid."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivada da Sigmoid, usada no cálculo do delta."""
        return x * (1 - x)

    def train(self, inputs, expected_output):
        # --- FORWARD PASS (Passagem Direta) ---

        # 1. Adiciona o bias à entrada e calcula a saída da camada oculta
        x = np.append(inputs, 1) # Adiciona o bias no final do vetor de entrada
        u_hidden = np.dot(x, self.wh)
        h = self._sigmoid(u_hidden)

        # 2. Adiciona o bias à saída da camada oculta e calcula a saída final
        h_bias = np.append(h, 1) # Adiciona o bias no final do vetor da camada oculta
        u_output = np.dot(h_bias, self.wo)
        out = self._sigmoid(u_output)

        # --- BACKWARD PASS (Retropropagação do Erro) ---

        # 1. Calcula o delta da camada de saída (DO)
        error = expected_output - out
        delta_output = error * self._sigmoid_derivative(out)

        # 2. Calcula o delta da camada oculta (DH)
        # Propaga o erro da saída para a camada oculta
        error_hidden = np.dot(delta_output, self.wo.T) 
        # Remove o delta associado ao bias da camada oculta, pois ele não recebe sinal de camadas anteriores
        delta_hidden = error_hidden[:-1] * self._sigmoid_derivative(h)

        # --- ATUALIZAÇÃO DOS PESOS ---
        
        # 1. Ajusta os pesos da camada de saída (wo)
        # np.outer cria a matriz de ajuste necessária
        self.wo += self.learning_rate * np.outer(h_bias, delta_output)
        
        # 2. Ajusta os pesos da camada de entrada (wh)
        self.wh += self.learning_rate * np.outer(x, delta_hidden)
        
        return out

    def execute(self, inputs):
        """Executa a rede (apenas o forward pass) sem treinar."""
        x = np.append(inputs, 1)
        u_hidden = np.dot(x, self.wh)
        h = self._sigmoid(u_hidden)

        h_bias = np.append(h, 1)
        u_output = np.dot(h_bias, self.wo)
        out = self._sigmoid(u_output)
        
        return out

def train_and_test(test_name, data, num_inputs, num_hidden, num_outputs):
    """
    Função para instanciar, treinar e testar o MLP.
    """
    print(f"### Treinando e Testando MLP: {test_name} ###")
    
    # Separa os dados em entradas (X) e saídas esperadas (y)
    X = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])

    mlp = MLP(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)

    # Treina a rede por 10.000 épocas
    for epoch in range(10000):
        epoch_error = 0.0
        for inputs, expected in zip(X, y):
            output = mlp.train(inputs, expected)
            epoch_error += np.sum(np.abs(expected - output))
        
        if (epoch + 1) % 1000 == 0:
            print(f"{(epoch + 1):>5}ª época, erro de aproximação: {epoch_error:.4f}")

    # Testa a rede após o treinamento
    print(f"\nResultados após o treinamento para {test_name}:")
    for inputs, expected in zip(X, y):
        result = mlp.execute(inputs)
        # Arredonda o resultado para a classificação final (0 ou 1)
        final_result = np.round(result)
        print(f"Entrada: {inputs} Saída: {final_result} Esperado: {expected}")
    print("########################################\n")


if __name__ == "__main__":
    # Definição do problema XOR
    DADOS_PORTA_XOR = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    # Para o MLP, o teste mais importante é o XOR.
    # Arquitetura: 2 entradas, 4 neurônios ocultos, 1 saída.
    train_and_test("Porta Lógica XOR", DADOS_PORTA_XOR, 2, 4, 1)