# PROJETOS-IF687

## IF867 - Introdução à Aprendizagem Profunda

### 1ª atividade prática

Discente(s): Bruno Antonio dos Santos Bezerra

Período: 7°

### Instruções e Requisitos
- Objetivo: Implementar e treinar um Multilayer Perceptron (MLP), inteiramente em [NumPy](https://numpy.org/doc/stable/) ou [Numba](https://numba.readthedocs.io/en/stable/index.html), sem o uso de bibliotecas de aprendizado profundo.
- A atividade pode ser feita em dupla.

### Tarefas

__Implementação (50%):__

- Construa um MLP com uma camada de entrada, pelo menos duas camadas ocultas e uma camada de saída.
- Implemente pelo menos duas funções de ativação diferentes para as camadas ocultas; use Sigmoid e Linear para a camada de saída.
- Implemente forward e backpropagation.
- Implemente um otimizador de sua escolha, adequado ao problema abordado.
- Implemente as funções de treinamento e avaliação.

__Aplicação (30%):__

  Teste se os seus modelos estão funcionando bem com as seguintes tarefas:
  - Regressão
  - Classificação binária

__Experimentação (20%):__

  Teste os seus modelos com variações na arquitetura, no pré-processamento, etc. Escolha pelo menos uma das seguintes opções:
  - Variações na inicialização de pesos
  - Variações na arquitetura
  - Implementação de técnicas de regularização
  - Visualização das ativações e gradientes

***Bônus:*** Implemente o MLP utilizando uma biblioteca de machine learning (ex.: [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/?hl=pt-br), [tinygrad](https://docs.tinygrad.org/), [Jax](https://jax.readthedocs.io/en/latest/quickstart.html)) e teste-o em uma das aplicações e em um dos experimentos propostos. O bônus pode substituir um dos desafios de aplicação ou experimentos feitos em NumPy, ou simplesmente somar pontos para a pontuação geral.

### Datasets recomendados:
Aqui estão alguns datasets recomendados, mas fica a cargo do aluno escolher os datasets que utilizará na atividade, podendo escolher um dataset não listado abaixo.
- Classificação

  - [Iris](https://archive.ics.uci.edu/dataset/53/iris)
  - [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
  - [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

- Regressão

  - [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality)
  - [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
  - [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

### Requisitos para Entrega

Um notebook Jupyter (de preferência, o link do colab) ou script Python contendo:

- Código: Implementação completa da MLP.
- Gráficos e Análises: Gráficos da curva de perda, ativações, gradientes e insights do treinamento, resultantes dos experimentos com parada antecipada e diferentes técnicas de regularização.
- Relatório: Um breve relatório detalhando o impacto de várias configurações de hiperparâmetros(ex.: inicialização de pesos, número de camadas ocultas e neurônios) e métodos de regularização no desempenho do modelo.
