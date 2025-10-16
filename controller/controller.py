#Aqui funções Controller de Cadastro (Categoria e Produtos)
from model import model
import re

#Categoria - OK
#Produto - OK
#Vendas - OK

class controller_cadastro:
    def __init__(self):
        self._model_db = model
        self.view = None
        self.app = None

    def set_view(self, view_instance):
        self.view = view_instance
    
    def set_app(self, app_instance):
        self.app = app_instance

    def mostrar_erro(self, mensagem: str):
        if self.app:
            self.app.exibir_mensagem_erro(mensagem)

    def mostrar_sucesso(self, mensagem: str):
        if self.app:
            self.app.exibir_mensagem_sucesso(mensagem)

    def limpar_campos(self):
        if self.app:
            self.app.limpar_campos_view()

    def cadastrarCategoria(self, descricao_categoria):           
        if not self.validacao_cad_categoria(descricao_categoria):
            return False
        
        if self._model_db.salvarCategorias(descricao_categoria.strip()):
            print("Controller: Categoria salva com sucesso via Model.")
            self.mostrar_sucesso(f"Categoria '{descricao_categoria.strip()}' cadastrada!")
            return True  
                
        else:
            validacao_cat_2 = f"Categoria {descricao_categoria} cadastrada com sucesso!"
            print(validacao_cat_2)
            self.mostrar_sucesso(validacao_cat_2)

    def cadastrarProduto(self, codigo_produto: int, codigobarras: int, descricao_produto: str, marca: str, categoria: str, custo: float, venda: float, curva: int):
        
        if not self.validacao_cad_produto(codigo_produto, codigobarras, descricao_produto, marca, categoria, custo, venda, curva):
            return False
        
        print(f"Cadastrando Produto: {descricao_produto}")
        if self._model_db.salvarProdutos(codigo_produto, codigobarras, descricao_produto.strip(), marca.strip(), categoria.strip(), custo, venda, curva):
            print("Controller: Produto salvo com sucesso via Model.")
            self.mostrar_sucesso(f"Produto '{descricao_produto.strip()}' cadastrado!")
            self.limpar_campos()
            return True
        else:
            print("Controller: Falha ao salvar produto via Model.")
            return False
    def cadastrarVenda(self, codigo_produto, quantidade_vendida, periodo_vendas):
        if not self.validacao_cad_vendas(codigo_produto, quantidade_vendida, periodo_vendas):
            return False
        if self._model_db.salvarVendas(codigo_produto, quantidade_vendida, periodo_vendas):
            validacao_ven_5 = f"Venda registrada com sucesso! Produto {codigo_produto}, Quantidade: {quantidade_vendida}, Período: {periodo_vendas} dias."
            print(validacao_ven_5)
            self.mostrar_sucesso(validacao_ven_5)
            return True
        else:
            return False
    ##### CADASTRO DE VENDA EM LOTES - SEM MENSAGEM
    def cadastrarVendaLote(self, codigo_produto, quantidade_vendida, periodo_vendas):
        if not self.validacao_cad_vendas(codigo_produto, quantidade_vendida, periodo_vendas):
            return False
        if self._model_db.salvarVendas(codigo_produto, quantidade_vendida, periodo_vendas):
            return True
        else:
            return False
    def cadastrarProdutoLote(self, codigo_produto: int, codigobarras: int, descricao_produto: str, marca: str, categoria: str, custo: float, venda: float, curva: int):
        if not self.validacao_cad_produto(codigo_produto, codigobarras, descricao_produto, marca, categoria, custo, venda, curva):
            return False
        if self._model_db.salvarProdutos(codigo_produto, codigobarras, descricao_produto.strip(), marca.strip(), categoria.strip(), custo, venda, curva):
            return True
        else:
            return False
    def processarCaminhoProdutos(self, arquivo_importado_p):
        if not arquivo_importado_p:
            print("Controller: Nenhum arquivo")
            return
        
        produtos_salvos = 0
        produtos_erro = 0
        lista_erros_p = []

        try:
            with open(arquivo_importado_p, mode='r', encoding='utf-8') as f:
                for numero_linha, linha_de_texto in enumerate(f, 1):
                    linha_limpa = linha_de_texto.strip()
                    if not linha_limpa:
                        continue

                    partes = linha_limpa.split(',')

                    try:
                        if len(partes) >= 8:
                            codigo_produto = int(partes[0])
                            codigobarras = int(partes[1])
                            descricao_produto = str(partes[2])
                            marca = str(partes[3])
                            categoria = str(partes[4])
                            custo = float(partes[5].replace('(', '').replace(')', '').replace(',', '.'))
                            venda = float(partes[6].replace('(', '').replace(')', '').replace(',', '.'))
                            curva = int(partes[7])

                            if self.cadastrarProdutoLote(codigo_produto, codigobarras, descricao_produto, marca, categoria, custo, venda, curva):
                                produtos_salvos += 1
                        else:
                            print(f"Aviso: Linha {numero_linha} ignorada. Formato Inválido: {linha_limpa}")
                            produtos_erro += 1
                            lista_erros_p.append(numero_linha)
                    except ValueError:
                        print(f"Aviso: Linha {numero_linha} ignorada. Contém dados não numéricos: {linha_limpa}")
                        produtos_erro += 1
                        lista_erros_p.append(numero_linha)

            if produtos_salvos > 0 and produtos_erro == 0:
                self.mostrar_sucesso(f"{produtos_salvos} produtos importadas com sucesso!")
            elif produtos_salvos > 0 and produtos_erro > 1:
                self.mostrar_sucesso(f"{produtos_salvos} venprodutosdas importados com sucesso!\n\n{produtos_erro} produtos ignorados - Linhas: {lista_erros_p}")
            else:
                self.mostrar_erro("Nenhum produto importado!")

        except FileNotFoundError:
            print(f"Erro no Controller: Arquivo não encontrado em '{arquivo_importado_p}'")
            self.mostrar_erro(f"Arquivo não encontrado no caminho especificado.")
        except Exception as e:
            print(f"Erro no Controller: {e}")
            self.mostrar_erro(f"Ocorreu um erro ao processar o arquivo.")
                            
    def processarCaminho(self, arquivo_importado):
        if not arquivo_importado:
            print("Controller: Nenhum arquivo")
            return

        vendas_salvas = 0
        vendas_erro = 0
        erros_lista = []

        try:
            with open(arquivo_importado, mode='r', encoding='utf-8') as f:
            
                for numero_linha, linha_de_texto in enumerate(f, 1):
                    linha_limpa = linha_de_texto.strip()
                    if not linha_limpa:
                        continue

                    partes = linha_limpa.split(',')
                    
                    try:
                        if len(partes) >= 3:
                            codigo_produto = int(partes[0])
                            qtde_vendida = int(partes[1])
                            periodo = int(partes[2])

                            #self.cadastrarVenda(codigo_produto, qtde_vendida, periodo)
                            if self.cadastrarVendaLote(codigo_produto, qtde_vendida, periodo):
                                vendas_salvas += 1
                        else:
                            print(f"Aviso: Linha {numero_linha} ignorada. Formato inválido: {linha_limpa}")
                            vendas_erro +=1
                            erros_lista.append(numero_linha)
                    except ValueError:
                        print(f"Aviso: Linha {numero_linha} ignorada. Contém dados não numéricos: {linha_limpa}")
                        vendas_erro += 1
                        erros_lista.append(numero_linha)
            
                        self.mostrar_sucesso("Arquivo importado e vendas cadastradas com sucesso!")
            ## Vendas > 0 sem erro      Vendas > 0 com erro
            if vendas_salvas > 0 and vendas_erro == 0:
                self.mostrar_sucesso(f"{vendas_salvas}  vendas importadas com sucesso!")
            elif vendas_salvas > 0 and vendas_erro > 1:
                self.mostrar_sucesso(f"{vendas_salvas} vendas importadas com sucesso!\n\n{vendas_erro} vendas ignoradas - Linhas: {erros_lista}")
            else:
                self.mostrar_erro("Nenhuma venda importada!")
        except FileNotFoundError:
            print(f"Erro no Controller: Arquivo não encontrado em '{arquivo_importado}'")
            self.mostrar_erro(f"Arquivo não encontrado no caminho especificado.")
        except Exception as e:
            print(f"Erro no Controller: {e}")
            self.mostrar_erro(f"Ocorreu um erro ao processar o arquivo.")

    def listarCategorias(self):
        return self._model_db.listarCategoriasModel()
    
    def validacao_cad_vendas(self, codigo_produto, quantidade_vendida, periodo_vendas):
        if not str(codigo_produto).strip() or not str(quantidade_vendida).strip() or not str(periodo_vendas).strip():
            validacao_ven_1 = "Preencha os campos vazios antes de prosseguir!"
            print(validacao_ven_1)
            self.mostrar_erro(validacao_ven_1)
            return False
        if not str(codigo_produto).isdigit():
            validacao_ven_2 = f"O valor digitado '{codigo_produto}' não é um código válido, utilize o código cadastrado!"
            print(validacao_ven_2)
            self.mostrar_erro(validacao_ven_2)
            return False
        if not str(quantidade_vendida).isdigit():
            validacao_ven_3 = f"A quantidade digitada '{quantidade_vendida}' não é válida, preencha corretamente antes de salvar!"
            print(validacao_ven_3)
            self.mostrar_erro(validacao_ven_3)
            return False
        if not str(periodo_vendas).isdigit():
            validacao_ven_4 = f"O período digitado '{periodo_vendas}' não é válido, preencha corretamente antes de salvar!"
            print(validacao_ven_4)
            self.mostrar_erro(validacao_ven_4)
            return False
        else:
            return True

    def validacao_cad_categoria(self, descricao_categoria: str):
        categorias_existentes = self._model_db.listarCategoriasModel()
        lista_formatada = [cat.lower() for cat in categorias_existentes]
        lista_limpa = descricao_categoria.strip()
        
        if not lista_limpa:
            validacao_cat_1 = "A descrição da categoria não pode ser vazia!"
            print(validacao_cat_1)
            self.mostrar_erro(validacao_cat_1)
            return False
        
        if descricao_categoria.lower() in list(lista_formatada):
            validacao_cat_3 = f"A categoria {descricao_categoria} já está cadastrada no banco!"
            print(validacao_cat_3)
            self.mostrar_erro(validacao_cat_3)
            return False
        
        if descricao_categoria.isdigit():
            validacao_cat_4 = f"Você digitou {descricao_categoria}, apenas números não são aceitos como nome de categorias, digite um nome válido!"
            print(validacao_cat_4)
            self.mostrar_erro(validacao_cat_4)
            return False
        if not re.search(r"[a-zA-Z0-9]", lista_limpa):
            validacao_cat_5 = "A categoria não pode ser composta apenas de caracteres especiais!"
            print(validacao_cat_5)
            self.mostrar_erro(validacao_cat_5)
        else:
            return True
    def validacao_cad_produto(self, codigo_produto: int, codigobarras: int, descricao_produto: str, marca: str, categoria: str, custo: float, venda: float, curva: int):
            campos_vazios = [codigo_produto, codigobarras, descricao_produto, marca, custo, venda, categoria, curva]
            categorias_existentes = self._model_db.listarCategoriasModel()
            lista_formatada = [cat.lower() for cat in categorias_existentes]

            codigos_existentes = [c[0] for c in self._model_db.listarCodigoProdutosModel()]
            print(campos_vazios)
            if not all(str(c).strip() for c in campos_vazios):
                validacao_pro_1 = "Existe campos sem dados no formulário, preencha os campos antes de salvar o produto!"
                print(validacao_pro_1)
                self.mostrar_erro(validacao_pro_1)
                return False
            if not str(codigo_produto).strip().isdigit():
                validacao_pro_10 = "O código de produto é inválido, o mesmo deve ser um número inteiro!"
                print(validacao_pro_10)
                self.mostrar_erro(validacao_pro_10)
                return False
            
            codigo_produto_n = int(codigo_produto)
            
            #print(codigo_produto)
            #print(codigos_existentes)

            if codigo_produto_n in codigos_existentes:
                validacao_pro_9 = f"O código '{codigo_produto_n}' já está cadastrado no sistema!"
                print(validacao_pro_9)
                self.mostrar_erro(validacao_pro_9)
                return False
            
            if not re.search(r"[a-zA-Z0-9]", descricao_produto):
                validacao_pro_6 = "A descrição não pode ser composta apenas de caracteres especiais!"
                print(validacao_pro_6)
                self.mostrar_erro(validacao_pro_6)
                return False

            if descricao_produto.isdigit():
                validacao_pro_5 = f"Não é permitido cadastro de produto com números apenas!"
                print(validacao_pro_5)
                self.mostrar_erro(validacao_pro_5)
                return False

            if not str(codigobarras).isdigit():
                validacao_pro_2 = "Código de Barras inválido!"
                self.mostrar_erro(validacao_pro_2)
                print(validacao_pro_2)
                return False
            
            try:
                custo = float(str(custo).replace(',', '.'))
            except ValueError as e:
                validacao_pro_3 = f"Erro {e}:\n Valor de custo {custo} inválido, digite novamente (Ex: 10.50 ou 10)"
                self.mostrar_erro(validacao_pro_3)
                print(validacao_pro_3)
                return False
            
            try:
                venda = float(str(venda).replace(',', '.'))
            except ValueError as e:
                validacao_pro_4 = f"Erro {e}:\n Valor de venda {venda} inválido, digite novamente (Ex: 10.50 ou 10)"
                self.mostrar_erro(validacao_pro_4)
                print(validacao_pro_4)
                return False
            
            if not isinstance(curva, (int)) or curva <= 0 or curva > 5:
                self.mostrar_erro("Insira a curva de vendas do Produto")
                return False
            
            if categoria == "Selecione a Categoria":
                validacao_pro_8 = "Selecione uma categoria válida!"
                print(validacao_pro_8)
                self.mostrar_erro(validacao_pro_8)
                return False

            if not categoria.lower() in list(lista_formatada):
                validacao_pro_7 = f"A categoria '{categoria}' não está cadastrada!"
                print(validacao_pro_7)
                self.mostrar_erro(validacao_pro_7)
                return False
            
            return True
    
    
    def obter_sugestao_de_compra(self, categoria, curva, periodo):
        print(f"Controller: Recebido pedido para Categoria={categoria}, Curva={curva}, Período={periodo}")
        resultados_do_banco = self._model_db.gerarSugestao(categoria, curva, periodo)

        return resultados_do_banco