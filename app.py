import customtkinter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from view.cadastro import CadastroFrame
from view.lista_usuarios import ListaUsuariosFrame
from view.dashboard import DashboardFrame
from view.components.menu import Menu

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x800")
        self.title("Sport Analiticy Br - Sistema Integrado")
        
        # Menu Principal
        self.menu = Menu(self, self.trocar_tela)
        
        self.container_telas = customtkinter.CTkFrame(self, fg_color="transparent")
        self.container_telas.pack(fill="both", expand=True, padx=10, pady=10)
        self.container_telas.grid_rowconfigure(0, weight=1)
        self.container_telas.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Inicialização
        for F in (CadastroFrame, ListaUsuariosFrame, DashboardFrame):
            nome = "cadastro" if F == CadastroFrame else "lista_usuarios" if F == ListaUsuariosFrame else "dashboard"
            frame = F(self.container_telas)
            self.frames[nome] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.trocar_tela("lista_usuarios")

    def trocar_tela(self, nome_tela):
        frame = self.frames[nome_tela]
        frame.tkraise()
        if nome_tela == "lista_usuarios":
            frame.configurar_tabela()

if __name__ == "__main__":
    os.system('cls')
    app = App()
    app.mainloop()