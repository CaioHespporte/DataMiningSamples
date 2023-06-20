import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

def main():
    input_file = '0-Datasets/transfusion-Clear.data'
    names = ['R','F','M','T','C']
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas 

    y = df['C']
    x = df.drop('C', axis = 1)

    x_treino, x_teste = x[0:374], x[374:]
    y_treino, y_teste = y[0:374], y[374:]
    
    # Criando a arquitetura da rede neural:
    modelo = Sequential()
    modelo.add(Dense(units=6, activation='relu', input_dim=x_treino.shape[1]))
    modelo.add(Dense(units=6, activation='relu'))
    modelo.add(Dense(units=1, activation='sigmoid'))

    # Treinando a rede neural:
    modelo.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    resultado = modelo.fit(x_treino, y_treino, epochs=200, batch_size=32, validation_data=(x_teste, y_teste))

    # Plotando gráfico do histórico de treinamento
    plt.plot(resultado.history['loss'])
    plt.plot(resultado.history['val_loss'])
    plt.title('Histórico de Treinamento')
    plt.ylabel('Função de custo')
    plt.xlabel('Épocas de treinamento')
    plt.legend(['Erro treino', 'Erro teste'])
    plt.show()
    
if __name__ == "__main__":
    main()
