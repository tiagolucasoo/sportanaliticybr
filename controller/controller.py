from model.elo_validacao import ValidacaoHandler
from model.elo_classificacao import ClassificacaoHandler
from model.elo_insercao import InsercaoHandler
from model.elo_dashboard import DashboardHandler

from model.model import ModelConsulta
import json

class ControllerAtleta:
    def __init__(self, view):
        self.view = view

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
        
        h1 = ValidacaoHandler()
        h2 = ClassificacaoHandler()
        h3 = DashboardHandler()
        h4 = InsercaoHandler()

        h1.set_proximo(h2).set_proximo(h3).set_proximo(h4)

        try:
            h1.processar(dados_atleta)
            self.view.mostrar_mensagem_status("Dados do atleta processados com sucesso.")
        except Exception as e:
            self.view.mostrar_mensagem_status(f"Erro: {e}")

        print("-" * 30)

    def salvar_esporte(self, esporte):
        print(f"Esporte recomendado: {esporte}")

    
    def buscar_atleta_por_nome(self, nome):
        model = ModelConsulta()
        return model.buscar_por_nome(nome)

    def buscar_todos_atletas(self):
        model = ModelConsulta()
        return model.buscar_todos()