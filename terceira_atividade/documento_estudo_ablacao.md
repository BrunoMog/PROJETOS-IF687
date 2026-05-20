# Documento de Estudo de Ablacao - Terceira Atividade

## 1. Objetivo
Este documento apresenta o estudo de ablacao realizado sobre a CNN implementada na atividade anterior (APS 2), utilizando o mesmo conjunto de dados (MNIST) e as mesmas metricas principais de avaliacao.

Foram conduzidos dois estudos:
1. Ablacao de regularizacao: com dropout vs sem dropout.
2. Ablacao arquitetural: 2 camadas convolucionais vs 3 camadas convolucionais.

## 2. Componente(s) escolhido(s)
### 2.1 Dropout (regularizacao)
Componente escolhido para o estudo principal: dropout nas camadas convolucionais e fully connected.

Comparacao:
1. Variante com dropout.
2. Variante sem dropout.

### 2.2 Numero de camadas convolucionais (ablacao adicional)
Para enriquecer a analise, foi adicionada uma segunda variacao focada na arquitetura da CNN:
1. Variante com 2 camadas convolucionais.
2. Variante com 3 camadas convolucionais.

## 3. Metodologia
1. Dataset: MNIST.
2. Framework: PyTorch.
3. Metricas analisadas: acuracia no teste, numero de parametros, FLOPs estimados, tempo de treino.
4. Ambiente: execucao local em CPU (com suporte a execucao rapida QUICK no notebook para validacao).
5. Artefatos gerados: modelos `.pth`, arquivos `.json` e `.csv`, e graficos no notebook.

## 4. Resultados
### 4.1 Ablacao de dropout
Fonte: `terceira_atividade/ablation_study_results.json`.

| Variante | Accuracy | Params | FLOPs | Tempo de treino (s) |
|---|---:|---:|---:|---:|
| with_dropout | 0.9849 | 2,363,130 | 3,974,528 | 1492.44 |
| without_dropout | 0.9857 | 2,363,130 | 3,974,528 | 272.06 |

Diferencas observadas (sem dropout em relacao a com dropout):
1. Accuracy: +0.0008 (aprox. +0.08 ponto percentual).
2. Params: sem alteracao.
3. FLOPs: sem alteracao.
4. Tempo de treino: -1220.38 s (aprox. 81.8% mais rapido).

### 4.2 Ablacao adicional de camadas convolucionais
Fonte: `terceira_atividade/ablation_conv_layers_results.csv`.

| Variante | Accuracy | Params | FLOPs | Tempo de treino (s) |
|---|---:|---:|---:|---:|
| 2_conv_layers | 0.9510 | 2,363,130 | 3,974,528 | 10.11 |
| 3_conv_layers | 0.9536 | 1,995,546 | 8,906,112 | 11.97 |

Diferencas observadas (3 camadas em relacao a 2 camadas):
1. Accuracy: +0.0026 (aprox. +0.26 ponto percentual).
2. Params: -367,584 (aprox. 15.6% menor).
3. FLOPs: +4,931,584 (aprox. 124.1% maior).
4. Tempo de treino: +1.86 s (aprox. 18.5% maior).

## 5. Opiniao sobre o impacto dos componentes
### 5.1 Impacto do dropout
Nesta configuracao, o dropout teve impacto pequeno (quase nulo) na acuracia e impacto negativo importante no tempo de treino. Isso sugere que, para este caso especifico, a regularizacao por dropout nao trouxe ganho de generalizacao que compense o custo computacional adicional.

### 5.2 Impacto da camada convolucional adicional
Adicionar uma terceira camada convolucional trouxe ganho pequeno de acuracia, com aumento relevante de FLOPs e tempo de treino. Portanto, a decisao de usar 3 camadas depende da prioridade:
1. Se a prioridade for maxima acuracia, 3 camadas pode ser aceitavel.
2. Se a prioridade for eficiencia computacional, 2 camadas e uma escolha mais equilibrada.

## 6. Conclusao
O estudo de ablacao mostra que nem todo componente considerado "essencial" produz ganho significativo em todas as configuracoes.

Resumo final:
1. Dropout: pouca diferenca de acuracia e maior custo de treino.
2. Camada conv adicional: pequeno ganho de acuracia, porem maior custo computacional.

Assim, para este projeto, uma arquitetura mais enxuta (sem dropout e/ou com 2 camadas convolucionais) pode oferecer melhor relacao entre desempenho e custo.

## 7. Link do repositorio
Repositorio: https://github.com/BrunoMog/PROJETOS-IF687.git

## 8. Arquivos de evidencia do estudo
1. `terceira_atividade/terceira_atividade.ipynb`
2. `terceira_atividade/ablation_study_results.json`
3. `terceira_atividade/ablation_study_results.csv`
4. `terceira_atividade/ablation_conv_layers_results.csv`
5. `terceira_atividade/models/best_with.pth`
6. `terceira_atividade/models/best_without.pth`
7. `terceira_atividade/models/best_2conv.pth`
8. `terceira_atividade/models/best_3conv.pth`
