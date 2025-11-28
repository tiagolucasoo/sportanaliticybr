import customtkinter

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        self.title("Sport Analiticy Br")

        self.containerDados01()
        
        self.containerTabela()
        self.configurar_tabela()

    def listLabels(self):
        return [
            "Consulta de Usuários", 
            "Arrumar", 
            "Arrumar", 
        ]

    def containerDados01(self):
        labels = self.listLabels()

        label01 = customtkinter.CTkLabel(self, text=labels[0])
        label01.pack(pady=(10, 2))



    def containerTabela(self):
        self.frame_resultados = customtkinter.CTkScrollableFrame(self, height=350, width=760)
        self.frame_resultados.pack(pady=10, padx=20)

    def configurar_tabela(self):
        for widget in self.frame_resultados.winfo_children():
            widget.destroy()

        colunas = ["Nome", "Esporte Sugerido", "Altura", "Peso", "Idade", "Salto V.", "Salto H.", "Arremesso", "Flexibilidade", "Resistência"]

        for i, texto_coluna in enumerate(colunas):
            label = customtkinter.CTkLabel(
                self.frame_resultados, 
                text=texto_coluna, 
                font=("Arial", 12, "bold"),
                width=10
            )
            label.grid(row=0, column=i, padx=5, pady=5)

    
    def button_callback(self):
        print("!")

if __name__ == "__main__":
    app = App()
    app.mainloop()