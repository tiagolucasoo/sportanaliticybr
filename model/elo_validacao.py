from model.elo_handler import HandlerElo

class ValidacaoHandler(HandlerElo):
    def processar(self, dados):
        print("[1/4] - Validando dados...")

        campos_obrigatorios = ["nome", "idade", "peso", "altura", "flexibilidade",
        "resistencia","arremesso", "salto_vertical", "salto_horizontal"]

        if not dados['nome']:
            raise Exception("O campo 'nome' não pode estar vazio.")

        for campo in campos_obrigatorios:
            if not dados.get(campo):
                raise Exception(f"O campo '{campo}' não foi preenchido")
        
        if dados['idade'] <= 0 or dados['peso'] <= 0 or dados['altura'] <= 0:
            raise Exception("Idade, peso e altura devem ser maiores que zero.")
        
        return super().processar(dados)