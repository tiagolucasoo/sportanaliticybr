import sqlite3, os
import model

def inserir_dados(nome, peso, altura, flexibilidade, resistencia, arremesso, salto_vertical, salto_horizontal):
    conn = model.rota_banco()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO usuario VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        nome, peso, altura, flexibilidade, resistencia, arremesso, salto_vertical, salto_horizontal)
        conn.commit()
        print(f"Dados do(a) {nome} Inseridos com sucesso")
        return True
    except sqlite3.IntegrityError as e:
        print(f"Erro de integridade: {e}")
        return False
    except Exception as ex:
        print(f"Erro ao salvar dados: {ex}")
        return False
    finally:
        conn.close()

inserir_dados("teste", 10, 10, 10, 10, 10, 10, 10)