import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import customtkinter
import controller.controller as controller

class ListaUsuariosFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.controller = controller.ControllerAtleta(self)

        self.container_header()
        self.container_tabela()
        self.configurar_tabela()

    def container_header(self):
        frame = customtkinter.CTkFrame(self, fg_color="transparent")
        frame.pack(pady=(20, 5), padx=20)

        titulo = customtkinter.CTkLabel(
            frame,
            text="Consulta de Atletas",
            font=("Arial", 22, "bold")
        )
        titulo.pack()

        subtitulo = customtkinter.CTkLabel(
            frame,
            text="Veja abaixo a lista de atletas cadastrados e seus resultados",
            font=("Arial", 14)
        )
        subtitulo.pack(anchor="w")

    def container_tabela(self):
        self.frame_resultados = customtkinter.CTkScrollableFrame(
            self,
            height=400,
            width=800,
            fg_color="#fff"
        )
        self.frame_resultados.pack(pady=20, padx=20, fill="both", expand=True)

    def configurar_tabela(self):
        for widget in self.frame_resultados.winfo_children():
            widget.destroy()

        colunas = ["Id", "Nome", "Altura", "Peso", "Idade", "Esporte Sugerido"]
        largura = [25, 300, 75, 75, 75, 200]

        for i, texto in enumerate(colunas):
            label = customtkinter.CTkLabel(
                self.frame_resultados,
                text=texto,
                font=("Arial", 14, "bold"),
                width=largura[i]
            )
            label.grid(row=0, column=i, padx=10, pady=8, sticky="w")
        
        atletas = self.controller.buscar_todos_atletas()
        for linha, atleta in enumerate(atletas, start=1):
            for coluna, valor in enumerate(atleta):
                label = customtkinter.CTkLabel(
                    self.frame_resultados,
                    text=str(valor),
                    font=("Arial", 12),
                    width=largura[coluna]
                )
                label.grid(row=linha, column=coluna, padx=10, pady=5, sticky="w")