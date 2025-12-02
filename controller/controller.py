from model.elo_validacao import ValidacaoHandler
from model.elo_classificacao import ClassificacaoHandler
from model.elo_insercao import InsercaoHandler
from model.elo_dashboard import DashboardHandler

from model.model import ModelConsulta
import json
import os

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
        dados_atleta = model.buscar_por_nome(nome)

        if not dados_atleta:
            return None

        id_db, nome_db, idade, peso, altura, flexibilidade, resistencia, arremesso, salto_vertical, salto_horizontal, esporte_recomendado, grafico_path = dados_atleta

        if grafico_path and os.path.exists(grafico_path):
            return dados_atleta

        try:
            print("[CONSULTA] Gráfico ausente — recalculando probabilidades com KNN e gerando gráfico...")
            dados_para_elo = {
                "nome": nome_db,
                "idade": idade,
                "peso": peso,
                "altura": altura,
                "flexibilidade": flexibilidade,
                "resistencia": resistencia,
                "arremesso": arremesso,
                "salto_vertical": salto_vertical,
                "salto_horizontal": salto_horizontal
            }

            h2 = ClassificacaoHandler()
            h3 = DashboardHandler()
            h2.set_proximo(h3)

            h2.processar(dados_para_elo)

            novo_grafico = dados_para_elo.get("grafico_path")
            novo_esporte = dados_para_elo.get("esporte_recomendado", esporte_recomendado)

            model.atualizar_grafico_path(nome_db, novo_grafico)

            lista = list(dados_atleta)
            lista[10] = novo_esporte
            lista[11] = novo_grafico

            return tuple(lista)

        except Exception as e:
            print(f"Erro ao regenerar gráfico na consulta: {e}")
            return dados_atleta
    
    def buscar_todos_atletas(self):
        model = ModelConsulta()
        return model.buscar_todos()