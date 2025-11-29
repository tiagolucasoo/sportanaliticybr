import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import customtkinter
import controller.controller as controller
from components.menu import containerMenu

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        self.title("Sport Analiticy Br")

        containerMenu(self, nome_pagina="Cadastro de Atletas")
        self.controller = controller.ControllerAtleta(self)
        self.containerDados01()
        self.containerDados02()
        self.containerBotao()
        self.atualizacaoBarraProgresso()

        #container_codigos = customtkinter.CTkFrame(App, fg_color="transparent")

    def listLabels(self):
        return [
            "Dados Biométricos", #Container01
            "Indicadores de Perfomance", #Container02
            "Rotina de Treino", #Container03
        ]

    def containerDados01(self):
        labels = self.listLabels()

        label01 = customtkinter.CTkLabel(self, text=labels[0])
        label01.pack(pady=(10, 2))

        subcontainer01 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer01.pack(side="top", pady=10)
        self.nome = customtkinter.CTkEntry(subcontainer01, placeholder_text="Nome", width=680)
        self.nome.pack(side="left", padx=20)
        
        subcontainer02 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer02.pack(side="top", pady=10)
        self.idade = customtkinter.CTkEntry(subcontainer02, placeholder_text="Idade", width=200)
        self.idade.pack(side="right", padx=20)
        self.altura = customtkinter.CTkEntry(subcontainer02, placeholder_text="Altura (cm)", width=200)
        self.altura.pack(side="left", padx=20)
        self.peso = customtkinter.CTkEntry(subcontainer02, placeholder_text="Peso (kg)", width=200)
        self.peso.pack(side="right", padx=20)

    def containerDados02(self):
        labels = self.listLabels()
        label02 = customtkinter.CTkLabel(self, text=labels[1])
        label02.pack(pady=(10, 2))

        subcontainer03 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer03.pack(side="top", pady=10)
        self.salto_vertical = customtkinter.CTkEntry(subcontainer03, placeholder_text="Salto Vertical (cm)", width=320)
        self.salto_vertical.pack(side="left", padx=20)
        self.salto_horizontal = customtkinter.CTkEntry(subcontainer03, placeholder_text="Salto Horizontal (cm)", width=320)
        self.salto_horizontal.pack(side="right", padx=20)

        subcontainer04 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer04.pack(side="top", pady=10)
        self.arremesso = customtkinter.CTkEntry(subcontainer04, placeholder_text="Arremesso (m)", width=200)
        self.arremesso.pack(side="left", padx=20)
        self.resistencia = customtkinter.CTkEntry(subcontainer04, placeholder_text="Abdominais (nº)", width=200)
        self.resistencia.pack(side="right", padx=20)
        self.flexibilidade = customtkinter.CTkEntry(subcontainer04, placeholder_text="Flexibilidade (cm)", width=200)
        self.flexibilidade.pack(side="right", padx=20)

    def mostrar_mensagem_status(self, mensagem):
        self.labelProgresso.configure(text=mensagem)

    def button_callback(self):
        try:
            nome = self.nome.get()
            idade = int(self.idade.get())
            altura = float(self.altura.get())
            peso = float(self.peso.get())
            salto_v = float(self.salto_vertical.get())
            salto_h = float(self.salto_horizontal.get())
            arremesso = float(self.arremesso.get())
            resistencia = int(self.resistencia.get())
            flexibilidade = float(self.flexibilidade.get())

            self.controller.salvar_atleta(nome, idade, altura, peso, salto_v, salto_h, arremesso, resistencia, flexibilidade)
        
        except ValueError as ve:
            print(f"Erro de valor: {ve}")
        
    def containerBotao(self):
        subcontainer05 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer05.pack(side="top", pady=10)

        salvar = customtkinter.CTkButton(subcontainer05, text="Salvar", command=self.button_callback, width=440)
        salvar.pack(side="left", padx=20, pady=20)

        limpar = customtkinter.CTkButton(subcontainer05, text="Limpar", command=self.button_callback, width=200)
        limpar.pack(side="right", padx=20, pady=20)

        self.progressbar = customtkinter.CTkProgressBar(self, orientation="horizontal", width=680, height=20, determinate_speed=1000)
        self.progressbar.pack()
        self.labelProgresso = customtkinter.CTkLabel(self, text="")
        self.labelProgresso.pack(pady=(10, 2))

    def atualizacaoBarraProgresso(self):
        lista = ["nome", "idade", "altura", "peso", "arremesso","resistencia", "flexibilidade", "salto_horizontal", "salto_vertical"]
    
        itens_preenchidos = 0
        total_itens = len(lista)

        for nome_do_widget in lista:
            try:
                widget = getattr(self, nome_do_widget)
                
                if widget.get() != "":
                    itens_preenchidos += 1
                    
            except AttributeError:
                print(f"Aviso: O widget '{nome_do_widget}' ainda não foi criado.")
            except Exception as e:
                print(f"Erro ao ler o widget '{nome_do_widget}': {e}")

        if total_itens > 0:
            progresso = itens_preenchidos / total_itens
        else:
            progresso = 0.0

        self.progressbar.set(progresso)
        self.labelProgresso.configure(text=f"Cadastro {round(progresso*100)} %")
        self.after(100, self.atualizacaoBarraProgresso)

if __name__ == "__main__":
    app = App()
    app.mainloop()