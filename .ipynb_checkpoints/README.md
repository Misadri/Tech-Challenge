# 🏥 Tech Challenge — Fase 1
## Sistema Inteligente de Suporte ao Diagnóstico — Saúde da Mulher

> Projeto desenvolvido para o **PosTech** como parte do Tech Challenge da Fase 1.  
> Aplicação de Machine Learning para classificação de risco de câncer de mama, com foco em apoio à decisão médica.

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Etapas do Notebook](#etapas-do-notebook)
- [Modelos Utilizados](#modelos-utilizados)
- [Métricas e Resultados](#métricas-e-resultados)
- [Explicabilidade](#explicabilidade)
- [Dashboard](#dashboard)
- [Discussão Crítica](#discussão-crítica)

---

## 📌 Sobre o Projeto

Uma rede de hospitais especializados no atendimento à mulher precisa de um sistema inteligente capaz de **apoiar profissionais de saúde na identificação precoce de condições de risco**.

Esta fase implementa a **base do sistema de IA com foco em Machine Learning**, analisando dados médicos automaticamente para identificar padrões de risco relacionados ao câncer de mama.

### Objetivos Técnicos

- Análise exploratória de dados médicos (EDA)
- Pré-processamento e pipeline de dados em Python
- Treinamento de 2 modelos de classificação
- Avaliação com métricas adequadas ao contexto médico
- Explicabilidade com Feature Importance e SHAP
- Dashboard visual de resultados

---

## 📊 Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

| Atributo | Valor |
|---|---|
| Fonte | UCI Machine Learning Repository / Kaggle |
| Link | [kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| Amostras | 569 |
| Features | 30 atributos numéricos |
| Classes | Maligno (212) / Benigno (357) |
| Valores ausentes | Nenhum |

As features descrevem características geométricas e texturais de núcleos celulares extraídos de imagens de biópsias por agulha fina (FNA), como raio, textura, perímetro, área, suavidade, compacidade, concavidade, entre outras — calculadas para a média, erro padrão e pior valor de cada amostra.

> O dataset já está disponível dentro do `scikit-learn` via `load_breast_cancer()`, sem necessidade de download manual.

---

## 🗂️ Estrutura do Projeto

```
tech-challenge-fase1/
│
├── tech_challenge.ipynb     # Notebook principal com toda a análise
├── dashboard.py             # Script para visualização em localhost
├── README.md                # Este arquivo
│
└── outputs/                 # Gerados automaticamente ao rodar o notebook
    ├── distribuicao_diagnosticos.png
    ├── distribuicao_features.png
    ├── correlacao.png
    ├── matrizes_confusao.png
    ├── comparacao_modelos.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── shap_beeswarm.png
```

---

## ⚙️ Pré-requisitos

- Python 3.8 ou superior
- pip

Verifique sua versão do Python:
```bash
python --version
```

---

## 🚀 Instalação

Clone o repositório e instale as dependências:

```bash
# Clone o repositório
git clone 
cd tech-challenge-fase1

# Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

Todas as bibliotecas necessárias em um único comando — nenhuma configuração adicional necessária.

---

## ▶️ Como Executar

### Opção 1 — Jupyter Notebook (análise completa)

```bash
jupyter notebook tech_challenge.ipynb
```

Execute as células sequencialmente com **Shift+Enter** ou clique em "Run All".

### Opção 2 — VS Code

1. Abra a pasta do projeto no VS Code
2. Instale a extensão **Jupyter** (Microsoft)
3. Abra o arquivo `tech_challenge.ipynb`
4. Selecione o kernel **Python 3**
5. Execute com **Shift+Enter** célula a célula

### Opção 3 — Dashboard Visual

Para visualizar os resultados de forma mais apresentável no navegador:

```bash
python dashboard.py
```

Abre automaticamente em `http://localhost:8000/dashboard.html`.  
Para encerrar, pressione **Ctrl+C** no terminal.

---

## 📓 Etapas do Notebook

### 1. Importação das Bibliotecas
Carregamento de todas as dependências: pandas, numpy, matplotlib, seaborn, scikit-learn e shap.

### 2. Carregamento e Exploração dos Dados (EDA)
- Carregamento do dataset via `sklearn.datasets.load_breast_cancer()`
- Visualização das primeiras linhas e shape
- Contagem e proporção das classes (Maligno vs Benigno)
- Estatísticas descritivas das features
- Verificação de valores ausentes
- Histogramas de distribuição das features principais por classe
- Mapa de calor de correlação entre features e target

### 3. Pré-processamento
- Separação de features (X) e target (y)
- Divisão treino/teste: **80% treino / 20% teste** com `stratify=y`
- Normalização via `StandardScaler` (fit apenas no treino, transform no teste)

### 4. Modelagem
- Treinamento de dois modelos: **Regressão Logística** e **Árvore de Decisão**
- Predições no conjunto de teste

### 5. Avaliação
- Acurácia, Recall e F1-Score para ambos os modelos
- `classification_report` completo
- Matrizes de confusão comparativas

### 6. Explicabilidade
- **Feature Importance** da Árvore de Decisão
- **SHAP Values** da Regressão Logística (summary plot e beeswarm)

### 7. Discussão Crítica
Análise dos resultados, limitações do modelo e considerações para uso em produção clínica.

---

## 🤖 Modelos Utilizados

### Regressão Logística
Modelo linear para classificação binária. Escolhido por:
- Alta interpretabilidade
- Boa performance em dados médicos com features correlacionadas
- Compatível com SHAP LinearExplainer para explicabilidade
- Excelente baseline para o problema

**Configuração:** `max_iter=1000`, `random_state=42`, dados normalizados com StandardScaler.

### Árvore de Decisão
Modelo baseado em regras de decisão. Escolhido por:
- Visualmente interpretável (estrutura de árvore)
- Não requer normalização dos dados
- Fornece Feature Importance nativa
- Complementa a Regressão Logística na análise comparativa

**Configuração:** `max_depth=5` (controla overfitting), `random_state=42`.

---

## 📈 Métricas e Resultados

### Por que Recall é a métrica prioritária?

No contexto de diagnóstico oncológico, os erros não têm o mesmo custo:

- **Falso Negativo** (maligno classificado como benigno): paciente não recebe tratamento → **consequência grave**
- **Falso Positivo** (benigno classificado como maligno): paciente realiza exames adicionais → **consequência aceitável**

Por isso, **maximizar o Recall da classe Maligno** é a estratégia mais segura para este problema.

### Resultados Obtidos

| Modelo | Acurácia | Recall (Maligno) | F1-Score (Maligno) |
|---|---|---|---|
| Regressão Logística | ~97% | ~97% | ~96% |
| Árvore de Decisão | ~93% | ~90% | ~91% |

> Os valores exatos são exibidos no notebook e no dashboard ao executar o projeto.

---

## 🔍 Explicabilidade

### Feature Importance (Árvore de Decisão)
Identifica globalmente quais features mais influenciam as decisões do modelo. As features com maior importância tendem a ser as primeiras divisões da árvore.

### SHAP (SHapley Additive exPlanations)
Técnica baseada em teoria dos jogos que explica a contribuição de **cada feature para cada predição individual**:

- **Valores positivos** → aumentam a probabilidade de ser Benigno
- **Valores negativos** → aumentam a probabilidade de ser Maligno
- **Cor vermelha** no beeswarm = valor alto da feature
- **Cor azul** = valor baixo da feature

As features mais determinantes identificadas: `worst concave points`, `worst perimeter` e `mean concave points` — alinhadas com o conhecimento clínico sobre morfologia tumoral maligna.

---

## 🖥️ Dashboard

O arquivo `dashboard.py` gera um dashboard web completo que exibe:

- Métricas dos dois modelos em cards visuais
- Distribuição dos diagnósticos no dataset
- Matrizes de confusão comparativas
- Gráfico de Feature Importance
- SHAP summary plot
- Discussão crítica dos resultados

Não requer instalação de bibliotecas adicionais além das já listadas.

---

## ⚠️ Discussão Crítica

### O modelo pode ser utilizado na prática?

**Sim, como ferramenta de apoio à triagem — com ressalvas importantes.**

✅ Casos de uso adequados:
- Priorização de casos urgentes em filas de espera
- Alertas para profissionais sobre padrões de alto risco
- Segunda opinião computacional em análises de rotina

❌ Limitações que impedem uso autônomo:
- Dataset pequeno (569 amostras) de fonte única
- Não inclui dados temporais ou histórico da paciente
- Não foi validado em dados externos ou de outras instituições
- Requer auditoria regular para detecção de viés

> **O médico sempre deve ter a palavra final no diagnóstico.**  
> Este sistema é uma ferramenta de suporte, nunca de substituição ao julgamento clínico.

---

## 📦 Dependências

| Biblioteca | Versão mínima | Uso |
|---|---|---|
| pandas | 1.3+ | Manipulação de dados |
| numpy | 1.21+ | Computação numérica |
| matplotlib | 3.4+ | Visualizações |
| seaborn | 0.11+ | Visualizações estatísticas |
| scikit-learn | 1.0+ | Modelos e métricas de ML |
| shap | 0.40+ | Explicabilidade |
| jupyter | 1.0+ | Execução do notebook |

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos — PosTech Tech Challenge Fase 1.

---

*Desenvolvido com Python 🐍 · scikit-learn · SHAP · PosTech 2025*
