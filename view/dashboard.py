import customtkinter

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        self.title("Sport Analiticy Br")
        
        self.containerPesquisaId()
    
    def listLabels(self):
        return [
            "Dados Biométricos", #Container01
            "Indicadores de Perfomance", #Container02
            "Rotina de Treino", #Container03
        ]
    
    def containerPesquisaId(self):
        labels = self.listLabels()

        label01 = customtkinter.CTkLabel(self, text=labels[0])
        label01.pack(pady=(10, 2))

        subcontainer01 = customtkinter.CTkFrame(self, fg_color="transparent")
        subcontainer01.pack(side="top", pady=10)
        #800
        self.nome = customtkinter.CTkEntry(subcontainer01, placeholder_text="Nome", width=500)
        self.nome.pack(side="left", padx=20)

        salvar = customtkinter.CTkButton(subcontainer01, text="Consultar", command=self.button_callback, width=300)
        salvar.pack(side="left", padx=20, pady=20)

    
    def button_callback(self):
        print("!")

if __name__ == "__main__":
    app = App()
    app.mainloop()