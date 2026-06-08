# Checklist de Atividades - Projeto 1

Este arquivo contém a lista de todas as tarefas necessárias para completar o
Projeto 1 (Análise de Sentimento). Use esta lista para acompanhar o progresso das
suas implementações em [project1.py](sentiment_analysis/project1.py) e
experimentos em [main.py](sentiment_analysis/main.py).

---

## 1. Funções de Perda Hinge

- [ ] Implementar a função `hinge_loss_single` em
  [project1.py](sentiment_analysis/project1.py) (perda hinge em um único ponto).
- [ ] Implementar a função `hinge_loss_full` em
  [project1.py](sentiment_analysis/project1.py) (média da perda hinge em todo o
  dataset).
- [ ] Validar localmente as funções de perda hinge executando os testes:
  ```bash
  python test.py
  ```

---

## 2. Algoritmo de Perceptron

- [ ] Implementar `perceptron_single_step_update` em
  [project1.py](sentiment_analysis/project1.py) (atualização de passo único).
- [ ] Implementar `perceptron` em [project1.py](sentiment_analysis/project1.py)
  (Perceptron completo iterando por $T$ épocas).
- [ ] Implementar `average_perceptron` em
  [project1.py](sentiment_analysis/project1.py) (Perceptron médio).
- [ ] Validar localmente os classificadores do Perceptron executando os testes:
  ```bash
  python test.py
  ```

---

## 3. Algoritmo Pegasos

- [ ] Implementar `pegasos_single_step_update` em
  [project1.py](sentiment_analysis/project1.py) (atualização do Pegasos para um
  único ponto).
- [ ] Implementar `pegasos` em [project1.py](sentiment_analysis/project1.py)
  (algoritmo Pegasos completo iterando por $T$ passos).
- [ ] Validar localmente as funções do Pegasos executando os testes:
  ```bash
  python test.py
  ```

---

## 4. Discussão dos Algoritmos (Análise 2D)

- [ ] Descomentar o bloco de código do problema 5 no arquivo
  [main.py](sentiment_analysis/main.py).
- [ ] Rodar o script principal para obter os valores de $\theta$ e $\theta_0$:
  ```bash
  python main.py
  ```
- [ ] Preencher as respostas de coeficientes dos gráficos para Perceptron,
  Perceptron Médio e Pegasos no arquivo [requirement.md](requirement.md).
- [ ] Testar a convergência empírica rodando por mais iterações e marcar a
  alternativa correta de convergência no arquivo [requirement.md](requirement.md).

---

## 5. Classificação e Acurácia

- [ ] Implementar `classify` em [project1.py](sentiment_analysis/project1.py)
  (classificação de pontos de dados com base no sinal da predição).
- [ ] Implementar `classifier_accuracy` em
  [project1.py](sentiment_analysis/project1.py) (treinamento e cálculo de acurácia
  no treino e validação).
- [ ] Validar as funções de classificação executando os testes:
  ```bash
  python test.py
  ```

---

## 6. Acurácia de Linha de Base (Baseline)

- [ ] Implementar a binarização de palavras em `extract_bow_feature_vectors` no
  arquivo [project1.py](sentiment_analysis/project1.py).
- [ ] Descomentar o bloco relevante de baseline no arquivo
  [main.py](sentiment_analysis/main.py).
- [ ] Executar o script principal para obter os resultados básicos dos modelos:
  ```bash
  python main.py
  ```
- [ ] Preencher as acurácias de validação iniciais dos três modelos no arquivo
  [requirement.md](requirement.md).

---

## 7. Ajuste de Parâmetros (Tuning)

- [ ] Descomentar a seção do problema 8 no arquivo
  [main.py](sentiment_analysis/main.py) (busca em grade coordenada para os
  parâmetros $T$ e $\lambda$).
- [ ] Executar a busca de parâmetros e obter os melhores hiperparâmetros:
  ```bash
  python main.py
  ```
- [ ] Registrar as acurácias e os melhores valores encontrados de $T$ e $\lambda$
  no arquivo [requirement.md](requirement.md).
- [ ] Executar a predição no conjunto de teste e salvar o valor final da acurácia
  de teste em [requirement.md](requirement.md).
- [ ] Chamar `utils.most_explanatory_word` em `main.py` para descobrir os 10
  unigramas mais explicativos positivos e preenchê-los no arquivo
  [requirement.md](requirement.md).

---

## 8. Engenharia de Características (Feature Engineering)

- [ ] **Remoção de Stop Words**:
  - [ ] Atualizar `bag_of_words` no arquivo
    [project1.py](sentiment_analysis/project1.py) para filtrar as palavras listadas
    no arquivo [stopwords.txt](sentiment_analysis/stopwords.txt).
  - [ ] Executar o algoritmo Pegasos com $T=25$ e $L=0.01$ no conjunto de dados e
    obter a acurácia de teste com as stop words removidas.
  - [ ] Registrar a nova acurácia no arquivo [requirement.md](requirement.md).
- [ ] **Características de Contagem**:
  - [ ] Atualizar `extract_bow_feature_vectors` no arquivo
    [project1.py](sentiment_analysis/project1.py) para usar as contagens brutas em
    vez de um indicador binário (0 ou 1).
  - [ ] Avaliar a acurácia de teste do Pegasos e salvá-la em
    [requirement.md](requirement.md).
- [ ] **Explorações Opcionais**:
  - [ ] Testar outras heurísticas (comprimento de texto, palavras em maiúsculas,
    embeddings, limiarização de frequência mínima) para melhorar a acurácia.
