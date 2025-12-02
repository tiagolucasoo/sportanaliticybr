import json
from model.elo_handler import HandlerElo
from services.controller_dashboard import ControllerDashboard

class DashboardHandler(HandlerElo):
    def __init__(self):
        super().__init__()
        self.dashboard_service = ControllerDashboard()

    def processar(self, dados):
        print("[3/4]: Gerando gráfico...")
        
        probs = json.loads(dados["valores_esportes"])
        
        caminho_grafico = self.dashboard_service.gerar_grafico(
            atleta_id=dados['nome'],
            basquete=probs.get('Basquete', 0),
            futebol=probs.get('Futebol', 0),
            handebol=probs.get('Handebol', 0),
            lutas=probs.get('Lutas', 0),
            natacao=probs.get('Natação', 0),
            volei=probs.get('Vôlei', 0),
            esporte=dados['esporte_recomendado']
        )
        
        dados["grafico_path"] = caminho_grafico
        
        return super().processar(dados)