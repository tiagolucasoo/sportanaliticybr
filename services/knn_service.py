import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from services.knn_maps import Volei, Futebol, Basquete, Lutas, Natacao, Handebol

class KnnService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.knn = None
        self.train_model()

    def tipo_esporte(self, classe_esporte, nome_esporte):
        esporte_data = {
            'Idade': classe_esporte.list_idade(),
            'Altura': classe_esporte.list_altura(),
            'Peso': classe_esporte.list_peso(),
            'Flexibilidade': classe_esporte.list_flexibilidade(),
            'Salto Horizontal': classe_esporte.list_horizontal(),
            'Salto Vertical': classe_esporte.list_vertical(),
            'Arremesso': classe_esporte.list_arremesso(),
            'Resistência': classe_esporte.list_resistencia(),
        }
        df = pd.DataFrame(esporte_data)
        df['Esporte'] = nome_esporte
        return df

    def train_model(self):
        lista_dataframes_esportes = [
            self.tipo_esporte(Futebol, "Futebol"),
            self.tipo_esporte(Volei, "Vôlei"),
            self.tipo_esporte(Basquete, "Basquete"),
            self.tipo_esporte(Lutas, "Lutas"),
            self.tipo_esporte(Natacao, "Natação"),
            self.tipo_esporte(Handebol, "Handebol")
        ]

        df = pd.concat(lista_dataframes_esportes, ignore_index=True)
        print(df.items())
        X_train_dados = df.drop('Esporte', axis=1)
        y_train_valor = df['Esporte']

        x_scaled = self.scaler.fit_transform(X_train_dados)
        y_encoded = self.label_encoder.fit_transform(y_train_valor)

        k = 11
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(x_scaled, y_encoded)
        print("Modelo KNN treinado com sucesso!")

    def prever_esporte(self, dados_atleta):

        dados_df = pd.DataFrame([{
            'Idade': dados_atleta['idade'],
            'Altura': dados_atleta['altura'],
            'Peso': dados_atleta['peso'],
            'Flexibilidade': dados_atleta['flexibilidade'],
            'Salto Horizontal': dados_atleta['salto_horizontal'],
            'Salto Vertical': dados_atleta['salto_vertical'],
            'Arremesso': dados_atleta['arremesso'],
            'Resistência': dados_atleta['resistencia'],
        }])

        dados_scaled = self.scaler.transform(dados_df)
        previsao_encoded = self.knn.predict(dados_scaled)
        previsao_esporte = self.label_encoder.inverse_transform(previsao_encoded)[0]

        esportes = self.label_encoder.classes_
        probabilidades = self.knn.predict_proba(dados_scaled)[0]
        esporte_probabilidade = {
            esporte: round(prob * 100, 2) for esporte,
            prob in zip(esportes, probabilidades )
        }

        print(f"Esporte previsto: {previsao_esporte}")
        print(f"Probabilidades: {esporte_probabilidade}")
        return previsao_esporte, esporte_probabilidade  
    