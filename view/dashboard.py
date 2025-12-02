import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import customtkinter
import controller.controller as controller

class DashboardFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.controller = controller.ControllerAtleta(self)
        self.containerPesquisaId()

        self.container_grafico = customtkinter.CTkFrame(self, fg_color="transparent")
        self.container_grafico.pack(pady=10)
    
    def listLabels(self):
        return [
            "Dashbard - Consulta Por Nome", #Container01
            "Indicadores de Perfomance", #Container02
        ]
    
    def estiloEntrys(self):
        entry_style = {
            "border_color": "#E0E0FF",
            "border_width": 2,
            "corner_radius": 5
        }
        return entry_style
    def containerPesquisaId(self):
        labels = self.listLabels()

        label_dashboard = customtkinter.CTkLabel(self,
                                         text=labels[0],
                                         text_color="#645DD7",
                                         font=customtkinter.CTkFont(size=16, weight="bold"))
        label_dashboard.pack(pady=(10, 2))

        subcontainer = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer.pack(side="top", pady=10)

        self.nome = customtkinter.CTkEntry(subcontainer, placeholder_text="Nome", width=500, **self.estiloEntrys())
        self.nome.pack(side="left", padx=20)

        salvar = customtkinter.CTkButton(subcontainer, text="Consultar", command=self.button_callback, width=150)
        salvar.pack(side="left", padx=10, pady=20)

        limpar = customtkinter.CTkButton(subcontainer, text="Limpar", command=self.limpar_campos, width=75)
        limpar.pack(side="right", padx=10, pady=20)

    def button_callback(self):
        nome = self.nome.get()
        self.limpar_resultados()
        
        resultado = self.controller.buscar_atleta_por_nome(nome)
        label_resultado = customtkinter.CTkLabel(
                                        self.container_grafico,
                                         text=
                                         f"""
                                            Nome: {resultado[1]}, Idade: {resultado[2]}, Peso: {resultado[3]}, Altura: {resultado[3]},
                                            Flexibilidade: {resultado[4]}, Resistência: {resultado[5]}, Arremesso: {resultado[6]},
                                            Salto Vertical: {resultado[7]}, Salto Horizontal: {resultado[8]}  """,
                                         text_color="#000000",
                                         font=customtkinter.CTkFont(size=14))
        label_resultado.pack(pady=10)
        print(f"Resultado da busca: {resultado}")

        grafico_path = resultado[11]

        if grafico_path and os.path.exists(grafico_path):
            from PIL import Image
            img = Image.open(grafico_path)            
            grafico_base = customtkinter.CTkImage(light_image=img, size=(600, 400))
            self.grafico_base = grafico_base
            grafico = customtkinter.CTkLabel(self.container_grafico, image=grafico_base, text="")
            grafico.pack()
        else:
            customtkinter.CTkLabel(
                self.container_grafico,
                text="Gráfico não disponível.",
                text_color="#FF0000"
            ).pack()

    def limpar_resultados(self):
        for widget in self.container_grafico.winfo_children():
            widget.destroy()
    
    def limpar_campos(self):
        self.nome.delete(0, customtkinter.END)
        self.limpar_resultados()