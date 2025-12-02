import customtkinter

class Menu(customtkinter.CTkFrame):
    def __init__(self, master, navegacao):
        super().__init__(master)
        self.navegacao = navegacao
        self.configure(fg_color="transparent",
                       border_color="#1F271B",
                       border_width=2,
                       corner_radius=5,
                       height=250)
        self.pack(side="top", fill="x",pady=10)
        self.widget()

    def estiloButtonsMenu(self):
        primario_size = {"height": 30, "corner_radius": 10}

        primario_color = {"fg_color": "#E0E0FF", "text_color": "#645DD7", "hover_color": "#fff"}
        secundario_color = {"fg_color": "#FFE0E0", "text_color": "#FF4242", "hover_color": "#fff"}
        return primario_size, primario_color, secundario_color
    
    def widget(self):
        styleButton = self.estiloButtonsMenu()

        titulo = customtkinter.CTkLabel(self, text="Menu de Navegação", width=200, font=customtkinter.CTkFont(size=20, weight="bold"))
        titulo.pack()

        btn_container = customtkinter.CTkFrame(self, fg_color="transparent")
        btn_container.pack(pady=10)
        button1 = customtkinter.CTkButton(btn_container, text="Cadastro de Atletas", width=180, **styleButton[0], **styleButton[1], command=lambda: self.navegacao("cadastro"))
        button1.pack(side="left", pady=10, padx=10)
        button2 = customtkinter.CTkButton(btn_container, text="Consulta Geral", width=180, **styleButton[0], **styleButton[1], command=lambda: self.navegacao("lista_usuarios"))
        button2.pack(side="left", pady=10, padx=10)
        button3 = customtkinter.CTkButton(btn_container, text="Dashboard", width=180, **styleButton[0], **styleButton[1], command=lambda: self.navegacao("dashboard"))
        button3.pack(side="left", pady=10, padx=10)
        button4 = customtkinter.CTkButton(btn_container, text="Sair", width=90, **styleButton[0], **styleButton[2], command=self.quit)
        button4.pack(side="left", pady=10, padx=10)