from model.model import ModelAtleta

class ControllerAtleta:
    def __init__(self, view):
        self.view = view
        self.model = ModelAtleta()

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
        
        print(f"Dados processados: {dados_atleta}")

        self.model.inserir_dados(dados_atleta)
        
        # Knn - Analise
        
        self.view.mostrar_mensagem_status("Sucesso! Controller e Model conectados.")
        print("-" * 30)

    def mostrar_erro(self, mensagem):
        print(f"ERRO: {mensagem}")