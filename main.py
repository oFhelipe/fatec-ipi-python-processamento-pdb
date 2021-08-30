import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#lendo o documento csv
data = pd.read_csv(r'data.csv')

# separando os dados em dados dependentes(features) e dados independentes(classes)
features = data.iloc[:,: -1].values
classes = data.iloc[:, -1].values

#iniciando o imputer"modelagem" falando que os valores NAN(faltantes) vai seguir a estrategia mean(média) de preenchimento
imputerModel = SimpleImputer(missing_values=np.nan, strategy="mean")

#aplicando a regra da modelagem
features[: , 2:-1] = imputerModel.fit_transform(features[: , 2:-1])


#iniciando e configurando o transformador de coluna usando o OneHotEncoder
columnTransformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)

# aplicando o transformador de coluna
features = columnTransformer.fit_transform(features)


#iniciando o encoder de label
labelEncoder = LabelEncoder()

#aplicando o encoder label para rotular a coluna
classes = labelEncoder.fit_transform(classes)

#fazendo a coluna windy de true e false virar 1 e 0 com o LabelEncoder
features[:,-1] = labelEncoder.fit_transform(features[:,-1])

#separando basses de features e classes em teste e treino
features_treinamento, features_teste, classes_treinamento, classes_teste = train_test_split(
    features, classes, 
    test_size=0.15, 
    random_state=1
)

#iniciando o escalador padrão para padronizar os valores desejados
standardScaler = StandardScaler()

# Aplicando o padronizador nas colunas desejadas nas features de treinamento
features_treinamento[:, 4:6] = standardScaler.fit_transform(features_treinamento[:, 4:6])

# Aplicando o padronizador treinado nas features de teste
features_teste[:, 4:6] = standardScaler.transform(features_teste[:, 4:6])
