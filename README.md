# PROJETOS-IF687

Repositório dos projetos e atividades práticas desenvolvidos na disciplina **IF687 / IF867 - Introdução à Aprendizagem Profunda** (CIn - UFPE, 2026.1).

O repositório reúne implementações em **PyTorch**, englobando desde os conceitos fundamentais de redes neurais artificiais (MLP) até arquiteturas convolucionais profundas (CNNs), técnicas de regularização, *transfer learning*, estudos de ablação, regressão de marcos faciais e o projeto principal de classificação médica de células sanguíneas (glóbulos brancos).

---

## 📁 Estrutura do Repositório

```text
PROJETOS-IF687/
├── primeira_atividade/
│   └── primeira_atividade.ipynb        # Implementação de MLP e otimização multiobjetivo (Iris)
├── segunda_atividade/
│   ├── segunda_atividade.ipynb         # CNN do zero vs. ResNet-18 (MNIST & CIFAR-10) + Feature Maps
│   ├── RepeatChannels.py               # Utilitário para adaptação de 1 para 3 canais
│   └── best_cnn_mnist.pth              # Checkpoint do melhor modelo CNN para MNIST
├── terceira_atividade/
│   ├── terceira_atividade.ipynb        # Notebook com análise e gráficos do estudo de ablação
│   ├── documento_estudo_ablacao.md     # Relatório técnico completo de ablação
│   ├── ablation_utils.py               # Funções de treino, avaliação e medição de FLOPs
│   ├── RepeatChannels.py               # Utilitário de canais para PyTorch
│   ├── run_ablation.py                 # Script orquestrador dos experimentos
│   ├── run_first_variant.py            # Execução da variante com Dropout
│   ├── run_second_variant.py           # Execução da variante sem Dropout
│   ├── ablation_study_results.json     # Métricas comparativas (Dropout)
│   ├── ablation_study_results.csv      # Tabela de resultados (Dropout)
│   ├── ablation_conv_layers_results.csv # Tabela de resultados (Camadas Convolucionais)
│   └── models/                         # Pesos salvos de cada variante testada
├── quarta_atividade/
│   ├── quarta_atividade.ipynb          # Regressão de Marcos Faciais no CelebA (CNN vs. ResNet-18)
│   └── landmark_utils.py               # Utilitários de normalização e desnormalização de landmarks
├── projeto_principal/
│   ├── main_executado.ipynb            # Notebook executado com EDA, treinos e análise crítica
│   ├── dataset_utils.py                # Dataset customizado RaabinDataset (lazy loading)
│   ├── execucao_log.txt                # Log de execução do pipeline
│   └── optuna_studies/                 # Banco SQLite e logs de trials do Optuna
├── requirements/
│   └── requirements.txt                # Dependências do projeto
└── README.md                           # Documentação geral do repositório
```

---

## 🧠 Atividades Desenvolvidas

### 1. Primeira Atividade: Multilayer Perceptron (MLP)

- **Localização:** `primeira_atividade/primeira_atividade.ipynb`
- **Dataset:** Iris (`sklearn.datasets.load_iris`)
- **Objetivo:** Implementação e avaliação de uma rede Multilayer Perceptron (MLP) configurável em PyTorch, acompanhada de otimização multiobjetivo de hiperparâmetros.

#### Destaques da Implementação:
- **Arquitetura Flexível:** Classe PyTorch parametrizável por número de camadas ocultas, neurônios por camada e funções de ativação (`ReLU`, `Tanh`, etc.).
- **Treinamento e Validação:** Separação estratificada (treino, validação e teste), função de perda `CrossEntropyLoss` e mecanismo de *early stopping* baseado em paciência.
- **Otimização Multiobjetivo com Optuna:** Busca automatizada para balancear dois objetivos conflitantes: **maximizar acurácia** e **minimizar custo computacional** (FLOPs estimados via `fvcore`).
- **Análise Visual:** Geração de curvas de loss/acurácia, importância de hiperparâmetros e fronteira de Pareto.
- **Desafio Opcional:** Projeção 2D das 4 características do dataset Iris via PCA e plotagem das fronteiras de decisão da rede treinada.

---

### 2. Segunda Atividade: Redes Convolucionais (CNN) e Transfer Learning

- **Localização:** `segunda_atividade/segunda_atividade.ipynb`
- **Módulos:** `segunda_atividade/RepeatChannels.py`, `segunda_atividade/best_cnn_mnist.pth`
- **Datasets:** MNIST e CIFAR-10
- **Objetivo:** Projetar e treinar uma CNN customizada do zero, aplicar busca de hiperparâmetros com Optuna e comparar seu desempenho com uma arquitetura consagrada pré-treinada (*fine-tuning* com ResNet-18).

#### Destaques da Implementação:
- **CNN Customizada:** Rede convolucional com camadas de convolução, pooling, batch normalization e dropout.
- **Otimização com Optuna:** Ajuste sistemático de quantidade de camadas convolucionais, canais de saída, tamanho de filtros (*kernel size*), taxa de dropout e taxa de aprendizado.
- **Transfer Learning (ResNet-18):** Adaptação do modelo pré-treinado no ImageNet substituindo a camada totalmente conectada final (`fc`). Uso da transformação customizada `RepeatChannels` para adaptar o MNIST (1 canal monocromático) para os 3 canais esperados pela ResNet.
- **Resultados Comparativos:**
  - **MNIST:**
    - CNN Customizada: **99.32%** de acurácia no teste
    - ResNet-18 (Fine-tuned): **96.70%** de acurácia no teste
  - **CIFAR-10:**
    - ResNet-18 (Fine-tuned): **80.08%** de acurácia no teste
    - CNN Customizada: **61.91%** de acurácia no teste
- **Desafio Opcional:** Extração e plotagem das ativações dos kernels (*feature maps*) da primeira camada convolucional (`conv_layers.0`) para visualizar quais representações e padrões visuais a rede aprendeu a extrair.

---

### 3. Terceira Atividade: Estudo de Ablação (Ablation Study)

- **Localização:** `terceira_atividade/terceira_atividade.ipynb`, `terceira_atividade/documento_estudo_ablacao.md`
- **Scripts:** `ablation_utils.py`, `run_ablation.py`, `run_first_variant.py`, `run_second_variant.py`
- **Dataset:** MNIST
- **Objetivo:** Realizar um estudo de ablação rigoroso sobre a CNN desenvolvida na atividade anterior, isolando componentes para mensurar seu real impacto em desempenho preditivo e custo computacional.

#### Estudos Conduzidos:
1. **Ablação de Regularização (Dropout):** Comparação da CNN com Dropout vs. Sem Dropout.
2. **Ablação Arquitetural (Camadas Convolucionais):** Comparação entre 2 camadas convolucionais vs. 3 camadas convolucionais.

#### Resumo dos Resultados:

| Experimento | Variante | Acurácia (Teste) | Parâmetros | FLOPs | Tempo de Treino (s) |
|---|---|---:|---:|---:|---:|
| **Regularização** | `with_dropout` | 0.9849 | 2.363.130 | 3.974.528 | 1492.44 |
| **Regularização** | `without_dropout` | **0.9857** | 2.363.130 | 3.974.528 | **272.06** (-81.8%) |
| **Arquitetura** | `2_conv_layers` | 0.9510 | 2.363.130 | **3.974.528** | **10.11** |
| **Arquitetura** | `3_conv_layers` | **0.9536** | 1.995.546 | 8.906.112 (+124.1%) | 11.97 (+18.5%) |

#### Conclusão Crítica:
- O **Dropout** não agregou ganho perceptível de generalização para este problema/arquitetura no MNIST, mas aumentou drasticamente o tempo de treino em CPU.
- A **3ª camada convolucional** trouxe um ganho marginal (+0.26 p.p. de acurácia), porém com custo computacional mais que dobrado em FLOPs (+124.1%). O estudo demonstrou que componentes clássicos nem sempre justificam seu custo e precisam ser validados empiricamente.

---

### 4. Quarta Atividade: Regressão de Marcos Faciais (Landmark Detection)

- **Localização:** `quarta_atividade/quarta_atividade.ipynb`
- **Módulos:** `quarta_atividade/landmark_utils.py`
- **Dataset:** CelebA (`torchvision.datasets.CelebA`, `target_type="landmarks"`)
- **Objetivo:** Implementar modelos de Deep Learning para resolver uma tarefa contínua de **regressão**: localização de 5 marcos faciais anatômicos (olho esquerdo, olho direito, nariz, canto esquerdo da boca e canto direito da boca — totalizando 10 coordenadas $(x, y)$).

#### Destaques da Implementação:
- **Normalização Espacial:** Criação de `normalize_landmarks` e `denormalize_landmarks` para mapear as coordenadas da resolução original (218x178) para o intervalo $[0, 1]$, facilitando a convergência das funções de perda.
- **Modelos Avaliados:**
  1. **CNN Customizada de Regressão:** Arquitetura própria otimizada com Optuna e treinada com `MSELoss`.
  2. **ResNet-18 Adaptada:** Modelo convolucional profundo pré-treinado com camada linear final ajustada para produzir as 10 coordenadas contínuas.
- **Resultados no Conjunto de Teste:**
  - **CNN Customizada:** Erro Médio Absoluto (MAE) = **0.0855**
  - **ResNet-18:** Erro Médio Absoluto (MAE) = **0.0301** (ganho expressivo em precisão na predição dos marcos faciais)
- **Visualização:** Renderização gráfica dos marcos faciais preditos plotados sobre os rostos dos conjuntos de validação e teste.

---

### 5. Projeto Principal: Classificação de Glóbulos Brancos (Raabin WBC)

- **Localização:** `projeto_principal/main_executado.ipynb`
- **Módulos:** `projeto_principal/dataset_utils.py`, `projeto_principal/optuna_studies/`
- **Dataset:** **Raabin WBC** (amostras de esfregaço de sangue para diagnóstico citológico de leucócitos).
  - Classes (5 tipos de glóbulos brancos): `Basophil`, `Eosinophil`, `Lymphocyte`, `Monocyte` e `Neutrophil`.
  - Splits: **Treino** (10.175 imagens), **Teste A** (4.339 imagens) e **Teste B** (2.119 imagens).
- **Objetivo:** Classificação multiclasse de alta precisão em imagens médicas microscópicas, enfrentando desafios reais de desbalanceamento de classes e robustez frente a mudanças de domínio dimensional (*distribution shift*).

#### Destaques da Implementação:
- **Carregamento Otimizado:** Classe `RaabinDataset` com *lazy loading* para evitar sobrecarga de memória RAM e suporte nativo a múltiplos *workers* de leitura.
- **Análise Exploratória (EDA):** Verificação de distribuições de classes, inspeção visual das células e cálculo empírico das médias e desvios-padrão dos canais RGB para normalização específica da base.
- **Estratégia de Balanceamento:** Aplicação de *undersampling* balanceado para tornar o treino computacionalmente viável e equilibrar as classes minoritárias (ex.: Basófilos).
- **Modelos Comparados:**
  1. **CNN Base** (arquitetura convolucional própria customizada).
  2. **ResNet-18** (*Transfer Learning* com pesos ImageNet).
  3. **MobileNetV3 Large** (*Transfer Learning* voltada para eficiência).
  4. **EfficientNet-B0** (*Transfer Learning* com escalonamento balanceado de profundidade/largura).
- **Otimização de Hiperparâmetros:** Utilização do Optuna com persistência em banco SQLite (`optuna_study.db`).

#### Resultados Obtidos (Macro F1-Score):

| Modelo | Teste A (Cenário Ideal) | Teste B (Cenário com *Shift*) |
|---|---:|---:|
| **CNN Base** | **87.74%** | 1.88% |
| **EfficientNet-B0** | 81.76% | 9.72% |
| **MobileNetV3 Large** | 81.05% | 13.91% |
| **ResNet-18** | 79.72% | **14.12%** |

#### Análise Crítica e Conclusões do Projeto:
- **Conjunto Teste A:** Quando as imagens mantêm a proporção e resolução padronizada (575x575) e todas as 5 classes estão presentes, os modelos apresentaram excelente poder discriminativo, com a CNN Base alcançando F1-Score de **87.74%** e as redes pré-treinadas superando os **80%**.
- **Conjunto Teste B (Sensibilidade ao Redimensionamento):** O Teste B possui imagens de resoluções e proporções heterogêneas e contém apenas duas das classes. A aplicação de redimensionamento rígido (*resize*) sem preservação da razão de aspecto (*aspect ratio*) distorceu a morfologia das células e do núcleo, gerando uma degradação severa no desempenho.
- **Recomendações e Trabalhos Futuros:** Para dados citológicos e médicos, o redimensionamento bruto não é suficiente; é indispensável o emprego de técnicas que preservem a geometria das células, como *padding* com razão de aspecto fixa (*aspect ratio preserving resize*) e políticas de *data augmentation* geométrica.

---

## 🛠️ Tecnologias e Bibliotecas

- **Linguagem:** Python 3.10+ / 3.14+
- **Deep Learning:** [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/)
- **Otimização de Hiperparâmetros:** [Optuna](https://optuna.org/)
- **Métricas e Pré-processamento:** [Scikit-learn](https://scikit-learn.org/), [Pillow (PIL)](https://python-pillow.org/)
- **Visualização de Dados:** [Matplotlib](https://matplotlib.org/), [Plotly](https://plotly.com/)
- **Análise de Complexidade Computacional:** [fvcore](https://github.com/facebookresearch/fvcore)

---

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/BrunoMog/PROJETOS-IF687.git
cd PROJETOS-IF687
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate   # No Linux / macOS
# ou: venv\Scripts\activate # No Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements/requirements.txt
```

### 4. Executar os experimentos

Você pode rodar qualquer uma das atividades via Jupyter Notebook ou executar os scripts de apoio diretamente:

- **Jupyter Notebook / JupyterLab:**
  ```bash
  jupyter notebook
  ```
  Navegue até a pasta da atividade desejada e abra o notebook correspondente:
  - `primeira_atividade/primeira_atividade.ipynb`
  - `segunda_atividade/segunda_atividade.ipynb`
  - `terceira_atividade/terceira_atividade.ipynb`
  - `quarta_atividade/quarta_atividade.ipynb`
  - `projeto_principal/main_executado.ipynb`

- **Scripts de Ablação da Terceira Atividade:**
  ```bash
  python terceira_atividade/run_ablation.py
  ```

---

## 👨‍💻 Autor

- **Bruno** ([@BrunoMog](https://github.com/BrunoMog))  
- Disciplina: IF687 / IF867 - Introdução à Aprendizagem Profunda (CIn - UFPE, 2026.1)
