import sqlite3
import os

class ModelAtleta:
    def __init__(self):
        self.criar_tabela()

    def rota_banco(self):
        caminho_banco = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database/app_data.db'))
        conn = sqlite3.connect(caminho_banco)
        return conn

    def criar_tabela(self):
        try:
            conn = self.rota_banco()
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usuario(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome VARCHAR(40),
                    idade INT NOT NULL,
                    peso INT NOT NULL,
                    altura INT NOT NULL,
                    flexibilidade INT NOT NULL,
                    resistencia INT NOT NULL,
                    arremesso DECIMAL(8,2) NOT NULL,
                    salto_vertical INT NOT NULL,
                    salto_horizontal INT NOT NULL,
                    esporte_recomendado VARCHAR(30),
                    grafico_path VARCHAR(255)
                )
            ''')
            print("Banco de Dados e Tabela de Usuários conectada\n")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Erro ao conectar ao banco de dados: {e}")

    def inserir_dados(self, dados_atleta):
        try:
            conn = self.rota_banco()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO usuario (nome, idade, altura, peso, salto_vertical, salto_horizontal,
                arremesso, resistencia, flexibilidade, esporte_recomendado, grafico_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    dados_atleta["nome"],
                    dados_atleta["idade"],
                    dados_atleta["altura"],
                    dados_atleta["peso"],
                    dados_atleta["salto_vertical"],
                    dados_atleta["salto_horizontal"],
                    dados_atleta["arremesso"],
                    dados_atleta["resistencia"],
                    dados_atleta["flexibilidade"],
                    dados_atleta["esporte_recomendado"],
                    dados_atleta["grafico_path"]
                )
            )
            conn.commit()
            print(f"Dados do(a) {dados_atleta['nome']} inseridos com sucesso no banco de dados.")
        except Exception as e:
            print(f"Erro ao inserir dados no banco de dados: {e}")
        finally:
            conn.close()
    
    def buscar_por_nome(self, nome):
        try:
            conn = self.rota_banco()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM usuario WHERE nome = ?', (nome,))
            resultado = cursor.fetchone()
            return resultado
        except Exception as e:
            print(f"Erro ao buscar dados no banco de dados: {e}")
            return None
        finally:
            conn.close()
