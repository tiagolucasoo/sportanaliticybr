import json
from model.elo_handler import HandlerElo
from services.knn_service import KnnService

class ClassificacaoHandler(HandlerElo):
    def __init__(self):
        super().__init__()
        self.knn_service = KnnService()

    def processar(self, dados):
        print("[2/4]: Classificando atleta (IA)...")
        
        esporte, probabilidades = self.knn_service.prever_esporte(dados)
        
        dados["esporte_recomendado"] = esporte
        dados["valores_esportes"] = json.dumps(probabilidades)
        
        print(f"   -> Esporte definido: {esporte}")
        return super().processar(dados)