import os
import requests
import json
import urllib.parse

class ControllerDashboard:
    def __init__(self):
        self.caminho_imagens = "./assets/images/dashboard/"

        if not os.path.exists(self.caminho_imagens):
            os.makedirs(self.caminho_imagens)
    
    def rota_imagem(self, basquete, futebol, handebol, lutas, natacao, volei, esporte):
        labels = ["Basquete", "Futebol", "Handebol", "Lutas", "Natação", "Vôlei"]
        valores = [basquete, futebol, handebol, lutas, natacao, volei]

        params = {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": f"Aptidão para {esporte}",
                    "data": valores,
                    "backgroundColor": "rgba(100, 93, 215, 0.2)",
                    "borderColor": "rgba(100, 93, 215, 1)",
                    "pointBackgroundColor": "rgba(100, 93, 215, 1)"
                }]
            },
            "options": {
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Desempenho por Esporte"
                    }
                },
                "scales": {
                    "r": {
                        "beginAtZero": True,
                        "min": 0,
                        "max": 100,
                        "grid": {
                            "color": "rgba(0,0,0,0.3)"
                        },
                        "angleLines": {
                            "color": "rgba(0,0,0,0.3)"
                        },
                        "pointLabels": {
                            "color": "#000",
                            "font": {
                                "size": 12
                            }
                        }
                    }
                }
            }
        }


        convert = json.dumps(params)
        encode = urllib.parse.quote(convert)
        url = f"https://quickchart.io/chart?c={encode}"
           
        return url
    
    def gerar_grafico(self, atleta_id: int,
                      basquete: int,
                      futebol: int,
                      handebol: int,
                      lutas: int,
                      natacao: int,
                      volei: int,
                      esporte: str):
        caminho_arquivo = os.path.join(self.caminho_imagens, f"{atleta_id}.png")

        if os.path.exists(caminho_arquivo):
            return caminho_arquivo
        
        url = self.rota_imagem(
            basquete=basquete,
            futebol=futebol,
            handebol=handebol,
            lutas=lutas,
            natacao=natacao,
            volei=volei,
            esporte=esporte
        )

        response = requests.get(url)

        if response.status_code == 200:
            with open(caminho_arquivo, 'wb') as f:
                f.write(response.content)
            return caminho_arquivo
        else:
            raise Exception("Falha ao gerar o gráfico.")