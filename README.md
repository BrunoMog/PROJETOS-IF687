# PROJETOS-IF687

Repositório dos projetos da disciplina IF867 - Introdução à Aprendizagem Profunda (T01, 2026.1).

## Primeira atividade: MLP

Implementação e avaliação de uma rede neural Multilayer Perceptron (MLP) usando PyTorch, com otimização de hiperparâmetros e visualizações de desempenho na base Iris.

## Estrutura do repositório

- `primeira_atividade/mlp_project.ipynb`: notebook principal da atividade.

## O que foi implementado no notebook

- Carregamento da base Iris (`sklearn.datasets.load_iris`).
- Split estratificado em treino, validação e teste.
- Classe de dataset customizada para PyTorch.
- MLP configurável por:
    - número de camadas ocultas;
    - número de neurônios por camada;
    - função de ativação.
- Treinamento com:
    - função de perda CrossEntropy;
    - early stopping por paciência;
    - métrica de acurácia.
- Otimização multiobjetivo com Optuna:
    - maximizar acurácia;
    - minimizar FLOPs.
- Visualizações:
    - importância de hiperparâmetros;
    - fronteira de Pareto;
    - curva de treino/validação (loss e métrica).
- Desafio opcional implementado:
    - fronteira de decisão em 2D via PCA (projeção da entrada 4D).

## Conferência de aderência aos requisitos da atividade

1. Implementação da MLP em PyTorch/TensorFlow:
Status: atendido.
Observação: a implementação foi feita em PyTorch.

2. Entrada com número de camadas, neurônios por camada, taxa de aprendizado e taxa de momento:
Status: atendido com ressalva de escopo.
Observação: camadas, neurônios e learning rate estão parametrizados. A taxa de momento não foi incluída na busca principal por decisão de projeto, pois diferentes otimizadores exigem conjuntos distintos de parâmetros.

3. Escolha do algoritmo otimizador como entrada do usuário (quando usar PyTorch/TensorFlow):
Status: atendido com configuração padrão fixa no experimento principal.
Observação: o código já aceita entrada de `optimizer_fn` e `optimizer_params` para parâmetros adicionais, mas a execução base foi mantida com Adam para padronizar a comparação entre testes.

4. Uso de base simples (ex.: Iris):
Status: atendido.

5. Escolha de hiperparâmetros por tentativa e erro:
Status: atendido.
Observação: a busca foi automatizada com Optuna.

6. Avaliação de desempenho:
Status: atendido.
Observação: inclui acurácia, curvas de loss/métrica e análise de custo computacional (FLOPs).

7. Desafio opcional (fronteira de decisão):
Status: atendido.

## Como executar

1. Crie e ative um ambiente virtual (recomendado).
2. Instale as dependências:

```bash
pip install torch scikit-learn matplotlib optuna fvcore plotly numpy
```

3. Abra e execute o notebook:

```bash
jupyter notebook primeira_atividade/mlp_project.ipynb
```
