import sys
import os
import time
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
        self.containerDadosFisicos()
        self.containerIndicadores()
        self.containerBotao()
        self.atualizacaoBarraProgresso()
    
    def listLabels(self):
        return [
            "Dados Físicos", #Container01
            "Indicadores", #Container02
        ]
    def estiloEntrys(self):
        entry_style = {
            "border_color": "#E0E0FF",
            "border_width": 2,
            "corner_radius": 5
        }
        return entry_style

    def containerDadosFisicos(self):
        labels = self.listLabels()

        label_fisico = customtkinter.CTkLabel(self, 
                                         text=labels[0],
                                         text_color="#645DD7",
                                         font=customtkinter.CTkFont(size=16, weight="bold"))
        label_fisico.pack(pady=(10, 2))

        subcontainer_nome = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer_nome.pack(side="top", pady=10)
        self.nome = customtkinter.CTkEntry(subcontainer_nome, placeholder_text="Nome", width=680, **self.estiloEntrys())
        self.nome.pack(side="left", padx=20)
        
        subcontainer_fisico = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer_fisico.pack(side="top", pady=10)
        self.idade = customtkinter.CTkEntry(subcontainer_fisico, placeholder_text="Idade", width=200, **self.estiloEntrys())
        self.idade.pack(side="right", padx=20)
        self.altura = customtkinter.CTkEntry(subcontainer_fisico, placeholder_text="Altura (cm)", width=200, **self.estiloEntrys())
        self.altura.pack(side="left", padx=20)
        self.peso = customtkinter.CTkEntry(subcontainer_fisico, placeholder_text="Peso (kg)", width=200, **self.estiloEntrys())
        self.peso.pack(side="right", padx=20)

    def containerIndicadores(self):
        labels = self.listLabels()
        label_indicador = customtkinter.CTkLabel(self,
                                         text=labels[1],
                                         text_color="#645DD7",
                                         font=customtkinter.CTkFont(size=16, weight="bold"))
        label_indicador.pack(pady=(10, 2))

        subcontainer_salto = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer_salto.pack(side="top", pady=10)
        self.salto_vertical = customtkinter.CTkEntry(subcontainer_salto, placeholder_text="Salto Vertical (cm)", width=320, **self.estiloEntrys())
        self.salto_vertical.pack(side="left", padx=20)
        self.salto_horizontal = customtkinter.CTkEntry(subcontainer_salto, placeholder_text="Salto Horizontal (cm)", width=320, **self.estiloEntrys())
        self.salto_horizontal.pack(side="right", padx=20)

        subcontainer_forca = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer_forca.pack(side="top", pady=10)
        self.arremesso = customtkinter.CTkEntry(subcontainer_forca, placeholder_text="Arremesso (m)", width=200, **self.estiloEntrys())
        self.arremesso.pack(side="left", padx=20)
        self.resistencia = customtkinter.CTkEntry(subcontainer_forca, placeholder_text="Abdominais (nº)", width=200, **self.estiloEntrys())
        self.resistencia.pack(side="right", padx=20)
        self.flexibilidade = customtkinter.CTkEntry(subcontainer_forca, placeholder_text="Flexibilidade (cm)", width=200, **self.estiloEntrys())
        self.flexibilidade.pack(side="right", padx=20)

    def mostrar_mensagem_status(self, mensagem):
        self.labelProgresso.configure(text=mensagem)

    def button_cadastro(self):
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
    
    def limpar_campos(self):
        self.nome.delete(0, customtkinter.END)
        self.idade.delete(0, customtkinter.END)
        self.altura.delete(0, customtkinter.END)
        self.peso.delete(0, customtkinter.END)
        self.salto_vertical.delete(0, customtkinter.END)
        self.salto_horizontal.delete(0, customtkinter.END)
        self.arremesso.delete(0, customtkinter.END)
        self.resistencia.delete(0, customtkinter.END)
        self.flexibilidade.delete(0, customtkinter.END)
        self.mostrar_mensagem_status("Campos limpos com sucesso.")
        
    def containerBotao(self):
        subcontainer05 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer05.pack(side="top", pady=10)

        salvar = customtkinter.CTkButton(subcontainer05, text="Salvar", command=self.button_cadastro, width=440)
        salvar.pack(side="left", padx=20, pady=20)

        limpar = customtkinter.CTkButton(subcontainer05, text="Limpar", command=self.limpar_campos, width=200)
        limpar.pack(side="right", padx=20, pady=20)

        self.progressbar = customtkinter.CTkProgressBar(self,
                                                        orientation="horizontal",
                                                        width=680,
                                                        height=20,
                                                        determinate_speed=1000,
                                                        fg_color="#E0E0FF",
                                                        progress_color="#FF0066")
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
        
        if progresso == 0:
            self.progressbar.configure(progress_color="#E0E0FF")
        elif progresso < 0.4:
            self.progressbar.configure(progress_color="#FF4242")
        elif progresso < 0.7:
            self.progressbar.configure(progress_color="#645DD7")
        else:
            self.progressbar.configure(progress_color="#00CC99")

        self.progressbar.set(progresso)
        self.labelProgresso.configure(text=f"Cadastro {round(progresso*100)} %")
        self.after(100, self.atualizacaoBarraProgresso)

if __name__ == "__main__":
    app = App()
    app.mainloop()