import os
import requests
import json

class ControllerDashboard:
    def __init__(self):
        self.caminho_imagens = "./assets/images/dashboard/"

        if not os.path.exists(self.caminho_imagens):
            os.makedirs(self.caminho_imagens)
    
    def rota_imagem(self, esportes):
        labels = list(esportes.keys())
        valores = list(esportes.values())

        params = {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Desempenho",
                    "data": valores
                }]
            }
        }
        url = f"https://quickchart.io/chart?c={json.dumps(params)}"
           
        return url
    
    def gerar_grafico(self, atleta_id: int, esportes: dict):
        caminho_arquivo = os.path.join(self.caminho_imagens, f"{atleta_id}.png")

        if os.path.exists(caminho_arquivo):
            return caminho_arquivo
        
        url = self.rota_imagem(esportes)
        response = requests.get(url)

        if response.status_code == 200:
            with open(caminho_arquivo, 'wb') as f:
                f.write(response.content)
            return caminho_arquivo
        else:
            raise Exception("Falha ao gerar o gráfico.")