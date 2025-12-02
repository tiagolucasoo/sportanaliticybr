from abc import ABC, abstractmethod

class HandlerElo(ABC):
    def __init__(self):
        self._proximo = None
    
    def set_proximo(self, handler):
        self._proximo = handler
        return handler
    
    @abstractmethod
    def processar(self, dados_atleta):
        if self._proximo:
            return self._proximo.processar(dados_atleta)
        return True