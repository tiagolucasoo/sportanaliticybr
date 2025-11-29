import customtkinter

def containerMenu(master, nome_pagina="Página Inicial"):

    def estiloButtonsMenu():
        primario_size = {"height": 30, "corner_radius": 10}

        primario_color = {"fg_color": "#E0E0FF", "text_color": "#645DD7", "hover_color": "#fff"}
        secundario_color = {"fg_color": "#FFE0E0", "text_color": "#FF4242", "hover_color": "#fff"}
        return primario_size, primario_color, secundario_color

    styleButton = estiloButtonsMenu()

    menu_container = customtkinter.CTkFrame(master, height=250, fg_color="transparent", border_color="#1F271B", border_width=3, corner_radius=10)
    menu_container.pack(side="top", pady=10)

    titulo = customtkinter.CTkLabel(menu_container, text="Menu de Navegação", width=200, font=customtkinter.CTkFont(size=20, weight="bold"))
    titulo.pack()
    mapa = customtkinter.CTkLabel(menu_container, text=f"Você está em: {nome_pagina}", font=customtkinter.CTkFont(size=14, slant="italic"))
    mapa.pack()

    button1 = customtkinter.CTkButton(menu_container, text="Cadastro de Atletas", width=180, **styleButton[0], **styleButton[1])
    button1.pack(side="left", pady=10, padx=10)
    button2 = customtkinter.CTkButton(menu_container, text="Consulta Geral", width=180, **styleButton[0], **styleButton[1])
    button2.pack(side="left", pady=10, padx=10)
    button3 = customtkinter.CTkButton(menu_container, text="Dashboard", width=180, **styleButton[0], **styleButton[1])
    button3.pack(side="left", pady=10, padx=10)
    button4 = customtkinter.CTkButton(menu_container, text="Sair", width=90, **styleButton[0], **styleButton[2])
    button4.pack(side="left", pady=10, padx=10)