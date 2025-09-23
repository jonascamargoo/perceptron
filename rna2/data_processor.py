import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Carrega, pré-processa e divide a base de dados Heart Disease.

    Responsabilidades:
    1. Busca o dataset da UCI.
    2. Trata valores ausentes (NaN) usando a média.
    3. Normaliza as features para o intervalo [0, 1].
    4. Converte o target para classificação binária (0 ou 1).
    5. Divide os dados em conjuntos de treino e teste.

    Retorna:
        (X_train, X_test, y_train, y_test)
    """
    print("1. Carregando e preparando a base de dados...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_imputed)

    y_binary = (y > 0).astype(int)
    y_reshaped = y_binary.values.reshape(-1, 1)

    print("Dados preparados com sucesso.")
    return train_test_split(X_normalized, y_reshaped, test_size=test_size, random_state=random_state)