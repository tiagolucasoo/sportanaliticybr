from model.elo_handler import HandlerElo
from model.model import ModelAtleta

class InsercaoHandler(HandlerElo):
    def __init__(self):
        super().__init__()
        self.model = ModelAtleta()

    def processar(self, dados):
        print("[4/4]: Persistindo no banco de dados...")
        
        sucesso = self.model.inserir_dados(dados)
        
        if not sucesso:
            raise Exception("Falha ao inserir no banco de dados.")
            
        print("   -> Ciclo concluído com sucesso.")
        return True