import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def main():

    input_file = '0-Datasets/transfusion-Clear.data'
    names = ['Recency', 'Frequency', 'Monetary', 'Time', 'C']
    features = ['R', 'F', 'M', 'T', 'C']
    target = 'C'
    df = pd.read_csv(input_file, names=names)
    
    
    #Distribuição de Frequência de doadores de sangue de março de 2007
    label = ['doaram', 'não doaram']
    cores = ['b', 'r']
    ndoaram = df['C'].value_counts()[0]
    doaram = df['C'].value_counts()[1]
    total = doaram + ndoaram
    y = np.array([doaram, ndoaram])
    plt.pie(y, labels=label, colors=cores,
            autopct=lambda x: '{:.0f}'.format(x*y.sum()/100, startangle=90))
    plt.title('Doadores')
  
    #Distribuição de Frequência de todas as classes
    g = sns.pairplot(df, hue = "C", palette="Set1", diag_kind="kde", height=2.5)
    
    plt.show()
   


if __name__ == "__main__":
    main()
