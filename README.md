# 🏅 Sport Analiticy Br - Sistema Integrado de Análise Esportiva
- Um sistema desktop desenvolvido para analisar características físicas de atletas e recomendar a modalidade esportiva mais adequada utilizando Inteligência Artificial. O foco é auxiliar na identificação de talentos com base em dados antropométricos.

## 📝 Criação do Projeto
- Desenvolvido como trabalho acadêmico utilizando a base de dados do **PROESP-Br**, seguindo a arquitetura **MVC (Model-View-Controller)** e implementando o padrão de projeto **Chain of Responsibility** para garantir um fluxo de processamento de dados robusto, desacoplado e escalável.

## ✨ Funcionalidades
* **Cadastro de Atletas:** Coleta de métricas físicas como altura, peso, envergadura, impulsão (salto vertical/horizontal) e flexibilidade.
* **Recomendação Inteligente:**
    * **Classificação KNN:** Utiliza o algoritmo *K-Nearest Neighbors* para comparar os dados do atleta com perfis já treinados.
    * **Predição:** Identifica a maior aptidão entre 6 modalidades: Futebol, Vôlei, Basquete, Lutas, Natação e Handebol.
* **Processamento em Cadeia:** O salvamento dos dados passa por uma "Esteira de Processamento" (Elos) que valida, classifica, gera gráficos e persiste os dados sequencialmente.
* **Dashboard Visual:**
    * **Gráfico de Radar:** Integração com API externa para gerar um *Spider Chart* comparativo, permitindo visualizar a aptidão do atleta em todas as modalidades simultaneamente.
* **Gestão de Dados:** Persistência local em banco de dados SQLite, com listagem para consulta posterior.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Interface:** CustomTkinter
* **Machine Learning:** Scikit-learn (KNN), Pandas, NumPy
* **Banco de Dados:** SQLite 3
* **API Gráfica:** QuickChart.io

## 📂 Estrutura do Projeto
-  app.py
- 📂 assets
- 📂 controller
  -  controller.py
- 📂 database
  -  app_data.db
- 📂 model
  - elo_classificacao.py
  - elo_dashboard.py
  - elo_handler.py
  - elo_insercao.py
  - elo_validacao.py
  - model.py
- 📂 services
   - controller_dashboard.py
   - knn_maps.py
   - knn_service.py
- 📂 view
   - 📂 components
       - menu.py
  - cadastro.py
  - dashboard.py
  - lista_usuarios.py
 
## 🚀 Como Executar
**Pré-requisitos**
_É necessário ter o **Python** instalado e conexão com a internet (para geração dos gráficos no momento do cadastro)._
1. **Clonar o Repositório**
```bash
   git clone https://www.github.com/tiagolucasoo/sportanaliticy.br
```
2. **Instalar Dependências**
```bash
   pip install customtkinter pandas numpy scikit-learn requests pillow
```
3. Executar a Aplicação Rode o arquivo principal na raiz do projeto:
```bash
python app.py
```
