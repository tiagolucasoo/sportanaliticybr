from model.model import ModelAtleta
from services.knn_service import KnnService
from services.controller_dashboard import ControllerDashboard
import json

class ControllerAtleta:
    def __init__(self, view):
        self.view = view
        self.model = ModelAtleta()
        self.knn_service = KnnService()
        self.dashboard_service = ControllerDashboard()

    def salvar_atleta(self, nome, idade, altura, peso, salto_v, salto_h, arremesso, resistencia, flexibilidade):
        print("-" * 30)
        print("CONTROLLER: Dados Recebidos da View")

        dados_atleta = {
            "nome": nome,
            "idade": idade,
            "altura": altura,
            "peso": peso,
            "salto_vertical": salto_v,
            "salto_horizontal": salto_h,
            "arremesso": arremesso,
            "resistencia": resistencia,
            "flexibilidade": flexibilidade
        }
        

        esporte_recomendado, probabilidades = self.knn_service.prever_esporte(dados_atleta)
        dados_atleta["esporte_recomendado"] = esporte_recomendado
        dados_atleta["valores_esportes"] = json.dumps(probabilidades)

        #Valores
        dict_esportes = json.loads(dados_atleta["valores_esportes"])
        dict_convert = list(dict_esportes.values())

        valor_basquete = dict_esportes['Basquete']
        valor_futebol = dict_esportes['Futebol']
        valor_handebol = dict_esportes['Handebol']
        valor_lutas = dict_esportes['Lutas']
        valor_natacao = dict_esportes['Natação']
        valor_volei = dict_esportes['Vôlei']

        print(f"Valores dos Esportes: {dict_convert}")
        print(valor_basquete)
        print(valor_futebol)

        grafico_path = self.dashboard_service.gerar_grafico(
            atleta_id=nome,
            basquete=valor_basquete,
            futebol=valor_futebol,
            handebol=valor_handebol,
            lutas=valor_lutas,
            natacao=valor_natacao,
            volei=valor_volei,
            esporte=esporte_recomendado
        )
        
        dados_atleta["grafico_path"] = grafico_path
        self.model.inserir_dados(dados_atleta)
        
        self.view.mostrar_mensagem_status("Sucesso! Controller e Model conectados.")
        print("-" * 30)

    def salvar_esporte(self, esporte):
        print(f"Esporte recomendado: {esporte}")

    def mostrar_erro(self, mensagem):
        print(f"ERRO: {mensagem}")
    
    def buscar_atleta_por_nome(self, nome):
        resultado = self.model.buscar_por_nome(nome)
        return resultado

    def buscar_todos_atletas(self):
        return self.model.buscar_todos()
