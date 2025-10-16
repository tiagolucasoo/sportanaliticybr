import sqlite3
import os

def rota_banco():
    caminho_banco = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database/app_data.db'))
    conn = sqlite3.connect(caminho_banco)
    return conn

def cadastro_usuario():
    conn = rota_banco()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuario(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome VARCHAR(40),
            peso INT NOT NULL,
            altura INT NOT NULL,
            flexibilidade INT NOT NULL,
            resistencia INT NOT NULL,
            arremesso DECIMAL(8,2) NOT NULL,
            salto_vertical INT NOT NULL,
            salto_horizontal INT NOT NULL
        )
    ''')
    print("Banco de Dados e Tabela de Usuários conectada\n")
    conn.commit()
    conn.close()

cadastro_usuario()