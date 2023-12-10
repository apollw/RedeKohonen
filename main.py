import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Rede_Kohonen:
    def __init__(self, input_size, output_size, sigma_0, tau1, eta_0, error_threshold):
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_0 = sigma_0
        self.tau1 = tau1
        self.eta_0 = eta_0
        self.error_threshold = error_threshold
        self.weights = np.random.rand(output_size, input_size)

    def train(self, data, plot_final=True):
        epoch = 0
        while epoch < 1000:  # Adicionado um limite máximo de épocas para evitar loop infinito
            total_error = 0

            for x in data:
                distances = np.linalg.norm(self.weights - x, axis=1)
                bmu_index = np.argmin(distances)
                total_error += distances[bmu_index]

                sigma_t = self.sigma_0 * np.exp(-epoch / self.tau1)
                h = np.exp(-0.5 * ((np.arange(self.output_size) - bmu_index) / sigma_t)**2)
                eta_t = self.eta_0 * np.exp(-epoch / self.tau1)
                delta_w = eta_t * h[:, np.newaxis] * (x - self.weights)
                self.weights += delta_w

            if total_error < self.error_threshold:
                break

            epoch += 1

        print(f"Treinamento concluído em {epoch} épocas.")
        print(f"Parâmetros: sigma_0={self.sigma_0}, tau1={self.tau1}, eta_0={self.eta_0}, error_threshold={self.error_threshold}")

        if plot_final:
            self.plot_kohonen(data, epoch)

    def predict(self, data):
        predictions = []
        for x in data:
            distances = np.linalg.norm(self.weights - x, axis=1)
            predictions.append(np.argmin(distances))
        return np.array(predictions)

    def plot_kohonen(self, data, epoch):
        plt.scatter(data[:, 0], data[:, 1], c=self.predict(data), cmap='viridis')
        plt.title(f'Final Result (Epoch {epoch})')
        plt.show()


# Carregando o conjunto de dados de dígitos
digits = datasets.load_digits()
data = digits.data

# Criando um dataframe com uma amostra reduzida
sample_size = 100
data_sample = digits.data[:sample_size]
target_sample = digits.target[:sample_size]

# Criando um dataframe usando pandas
df = pd.DataFrame(data_sample, columns=[f"pixel_{i}" for i in range(data_sample.shape[1])])
df["target"] = target_sample

# Descrição dos dados do Dataset
print(df.head())
print(df.describe())
print(df['target'].value_counts())

# Gráfico de Distribuição dos Dígitos
plt.figure(figsize=(8, 6))
df['target'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição dos Dígitos')
plt.xlabel('Dígito')
plt.ylabel('Contagem')
plt.show()

#Visualização gráfica de cada dígito do datasets
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.reshape(data_sample[i], (8, 8)), cmap='gray')
    ax.set_title(f"Label: {target_sample[i]}")
    ax.axis('off')
plt.show()

# Normalizando os dados
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Criando e treinando a rede de Kohonen
print("Treinando a rede...")
kohonen = Rede_Kohonen(input_size=data.shape[1], output_size=10, sigma_0=1.0, tau1=100, eta_0=0.1, error_threshold=0.01)
kohonen.train(data_normalized, plot_final=True)
print("Resultados...")

# Obtendo as ativações dos neurônios para cada amostra
activations = kohonen.predict(data_normalized)

# Obtendo os pesos da rede
weights = kohonen.weights

# Visualizando os protótipos como imagens
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.reshape(weights[i], (8, 8)), cmap='gray')
    ax.set_title(f"Neurônio {i}")
    ax.axis('off')
plt.show()

# Criando um gráfico de barras da média de ativações por classe
mean_activations = df.groupby('target')['target'].count().sort_index()
mean_activations.index = [f'Dígito {i}' for i in mean_activations.index]

plt.bar(mean_activations.index, mean_activations.values)
plt.title('Média de Ativações por Classe')
plt.xlabel('Dígito')
plt.ylabel('Média de Ativações')
plt.show()

from sklearn.decomposition import PCA

# Reduzir a dimensionalidade para 2D usando PCA
pca = PCA(n_components=2)
map_representation_pca = pca.fit_transform(weights)

# Visualizando o mapa em 2D
plt.figure(figsize=(10, 8))
plt.scatter(map_representation_pca[:, 0], map_representation_pca[:, 1], c=np.arange(10), cmap='viridis', s=100)
plt.title('Representação 2D do Mapa de Kohonen (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Classe')
plt.show()
