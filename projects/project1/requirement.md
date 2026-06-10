# 1. Introdução

O objetivo deste projeto é desenvolver um classificador para usar na análise de
sentimento de avaliações de produtos. Nosso conjunto de treinamento consiste em
avaliações escritas por clientes da Amazon para vários produtos alimentícios.
As avaliações, originalmente atribuídas em uma escala de 5 pontos, foram
ajustadas para uma escala de +1 ou -1, representando uma avaliação positiva
ou negativa, respectivamente.

Abaixo estão duas entradas de exemplo do nosso conjunto de dados. Cada entrada
consiste na avaliação e no seu rótulo. As duas avaliações foram escritas por
clientes diferentes, descrevendo a experiência deles com um doce sem açúcar.

| Avaliação | rótulo |
| :--- | :---: |
| *Nojento, sem sabor. O doce é apenas vermelho, sem sabor. Apenas simples e mastigável. Eu nunca os compraria novamente* | -1 |
| *DELICIOSO! Você nunca imaginaria que eles são sem açúcar e é tão ótimo que você pode comê-los praticamente sem culpa! Fiquei tão impressionado que encomendei alguns para mim mesmo (com chocolate amargo) para levar ao escritório. Estes são simplesmente EXCELENTES!* | 1 |

Para analisar automaticamente as avaliações, você precisará completar as
seguintes tarefas:

1. Implemente e compare três tipos de classificadores lineares: o algoritmo
   [perceptron](#), o algoritmo [perceptron médio](#) e o algoritmo
   [Pegasos](#).
2. Utilize seus classificadores no conjunto de avaliações alimentícias, usando
   algumas características de texto simples.
3. Experimente características adicionais e explore o impacto delas no
   desempenho do classificador.

---

## Detalhes da Configuração

Para este projeto e ao longo do curso, utilizaremos Python 3.11 com algumas
bibliotecas adicionais. Recomendamos fortemente que você observe como a
biblioteca numérica NumPy é utilizada no código fornecido e leia o tutorial
online do NumPy. **Os arrays do NumPy são muito mais eficientes do que os arrays
nativos do Python ao realizar computações numéricas. Além disso, usar o NumPy
reduzirá substancialmente as linhas de código que você precisará escrever.**

1. *Observação sobre o software: Para este projeto, você precisará da caixa de
   ferramentas numérica **NumPy** e da caixa de ferramentas de plotagem
   **matplotlib**.*

Baixe `sentiment_analysis.tar.gz` e descompacte-o em um diretório de trabalho.
A pasta `sentiment_analysis` contém os vários arquivos de dados no formato .tsv,
juntamente com os seguintes arquivos Python:

- [project1.py](sentiment_analysis/project1.py) contém várias funções úteis e
  modelos de funções que você utilizará para implementar seus algoritmos de
  aprendizagem.
- [main.py](sentiment_analysis/main.py) é um esqueleto de script onde essas
  funções são chamadas e você pode executar seus experimentos.
- [utils.py](sentiment_analysis/utils.py) contém funções utilitárias que a
  equipe implementou para você.
- [test.py](sentiment_analysis/test.py) é um script que executa testes em alguns
  dos métodos que você irá implementar. Note que esses testes são fornecidos para
  ajudá-lo a depurar sua implementação e não são necessariamente representativos
  dos testes usados para a avaliação online. Sinta-se à vontade para adicionar
  mais casos de teste localmente para validar ainda mais a corretude de seu
  código antes de submetê-lo aos avaliadores online nas caixas de código.

> [!TIP]
> Durante todo o sistema de avaliação online, você pode assumir que a
> biblioteca Python NumPy já está importada como np. Em alguns problemas, você
> também terá acesso à biblioteca Python random e a outras funções que você
> já implementou.

Este projeto se desenvolverá tanto no MITx quanto em sua máquina local. Você
pode implementar funções localmente e executar o `test.py` para validar a
funcionalidade básica, e então copiar e colar seu código nas caixas de código
do MITx para verificar completamente a corretude e receber sua nota por
implementações individuais de funções. Alternativamente, você também pode
implementar as funções online primeiro e, após finalizar, copiar e colar a
solução para o seu arquivo local [project1.py](sentiment_analysis/project1.py).
Cuidado com o número de tentativas que você tem para cada problema,
especialmente se você escolher o segundo fluxo de desenvolvimento.

### Como Testar Localmente

No seu terminal, navegue até o diretório onde seus arquivos do projeto estão
localizados. Execute o comando abaixo para rodar todos os testes disponíveis:

```bash
python test.py
```

### Como Executar suas Funções do Projeto 1 Localmente

No seu terminal, digite o comando abaixo. Você precisará descomentar/comentar o
código relevante no arquivo à medida que avança no projeto:

```bash
python main.py
```

> [!NOTE]
> Você também pode passar pela recapitulação no final desta unidade antes ou
> simultaneamente com este projeto.

# 2. Função de Perda Hinge

Neste projeto, você implementará classificadores lineares começando com o
algoritmo Perceptron. Você começará escrevendo sua função de perda, uma função de
perda hinge. Para essa função, são fornecidos os parâmetros do seu modelo
$\theta$ e $\theta_0$. Além disso, é fornecida uma matriz de características na
qual as linhas são vetores de características e as colunas são características
individuais, e um vetor de rótulos representando o sentimento real do vetor de
características correspondente.

---

## Hinge Loss em uma amostra de dados

Primeiro, implemente o cálculo básico da função de perda hinge em um único ponto
de dados. Em vez de toda a matriz de características, você recebe uma linha,
representando o vetor de características de uma amostra de dados única, e seu
rótulo de $+1$ ou $-1$ que representa o sentimento verdadeiro da amostra.

A função de perda hinge para um único ponto de dados $(x, y)$ é dada por:

$$L(y, \theta \cdot x + \theta_0) = \max(0, 1 - y(\theta \cdot x + \theta_0))$$

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def hinge_loss_single(feature_vector, label, theta, theta_0):
    # Seu código aqui
```

---

## Hinge Loss completa sobre o dataset

Em seguida, implemente a perda hinge média para os parâmetros de classificação
dados sobre um conjunto de dados completo (matriz de características e vetor de
rótulos). A perda hinge completa é a média das perdas hinge individuais de todos
os pontos de dados no conjunto.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    # Seu código aqui
```

# 3. Algoritmo de Perceptron

Agora você implementará o algoritmo Perceptron. O classificador linear
Perceptron tenta encontrar um hiperplano separador definido por $\theta$ e
$\theta_0$.

---

## Atualização de Passo Único do Perceptron

Implemente a atualização de um único passo do algoritmo Perceptron para um único
ponto de dados. Se o ponto de dados $(x, y)$ for classificado incorretamente ou
estiver na margem (ou seja, se $y(\theta \cdot x + \theta_0) \le 0$), atualize os
parâmetros da seguinte forma:

$$\theta \leftarrow \theta + y x$$
$$\theta_0 \leftarrow \theta_0 + y$$

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    # Seu código aqui
```

---

## Algoritmo Perceptron Completo

Implemente o algoritmo Perceptron completo sobre um conjunto de dados. Ele deve
rodar por $T$ iterações através do conjunto de dados (sem parada antecipada). Em
cada iteração, o algoritmo deve iterar sobre os pontos de dados na ordem
especificada por `get_order(feature_matrix.shape[0])`.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def perceptron(feature_matrix, labels, T):
    # Seu código aqui
```

---

## Algoritmo do Perceptron Médio

O Perceptron Médio adicionará uma modificação ao algoritmo perceptron original:
uma vez que o algoritmo básico continua atualizando à medida que é executado,
empurrando os parâmetros em direções possivelmente conflitantes, é melhor obter
uma média desses parâmetros como resposta final. Cada atualização do algoritmo é
a mesma de antes. Os parâmetros retornados, contudo, são uma média dos $\theta$
e $\theta_0$ ao longo dos $nT$ passos de atualização:

$$\theta_{\text{final}} = \frac{1}{nT} \left( \theta^{(1)} + \theta^{(2)} + \dots + \theta^{(nT)} \right)$$
$$\theta_{0, \text{final}} = \frac{1}{nT} \left( \theta_0^{(1)} + \theta_0^{(2)} + \dots + \theta_0^{(nT)} \right)$$

Acompanhar a média móvel através de loops pode ser difícil; acumular uma soma
através de loops e depois dividir por $nT$ no final é mais simples.

> [!WARNING]
> Certifique-se de chamar `get_order(feature_matrix.shape[0])` para iterar a
> matriz de características em cada época $t \in \{0, \dots, T-1\}$.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def average_perceptron(feature_matrix, labels, T):
    # Seu código aqui
```

# 4. Algoritmo Pegasos

Agora você implementará o algoritmo Pegasos. O Pegasos é um algoritmo de
subgradiente estocástico para resolver o problema de otimização do SVM (Support
Vector Machine).

O pseudo-código a seguir descreve a regra de atualização do Pegasos para um ponto
de dados $x^{(i)}$ com rótulo $y^{(i)}$, parâmetro de regularização $\lambda$ e
taxa de aprendizado decrescente $\eta$:

Se $y^{(i)}(\theta \cdot x^{(i)}) \le 1$, então a regra de atualização é:

$$\theta \leftarrow (1 - \eta \lambda) \theta + \eta y^{(i)} x^{(i)}$$

Caso contrário (se $y^{(i)}(\theta \cdot x^{(i)}) > 1$):

$$\theta \leftarrow (1 - \eta \lambda) \theta$$

Neste problema, você precisará adaptar essa regra de atualização para adicionar
um termo de viés ($\theta_0$) à hipótese, mas tome cuidado para não penalizar a
magnitude de $\theta_0$. Ou seja, $\theta_0$ não deve sofrer decaimento por
regularização $\lambda$, sendo atualizado apenas quando $y^{(i)}(\theta \cdot x^{(i)} + \theta_0) \le 1$ da seguinte forma:

$$\theta_0 \leftarrow \theta_0 + \eta y^{(i)}$$

---

## Atualização de Passo Único do Pegasos

Implemente a atualização de passo único do Pegasos para um único ponto de dados.
Esta função é muito semelhante à atualização do Perceptron, exceto pelo fato de
que ela deve utilizar as regras do Pegasos descritas acima. Ela receberá os
valores de $\lambda$ (parâmetro `L`) e $\eta$ (parâmetro `eta`).

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0):
    # Seu código aqui
```

---

## Algoritmo Pegasos Completo

Finalmente, implemente o algoritmo completo do Pegasos. Você receberá a matriz de
características, os rótulos, o número máximo de iterações $T$ e o parâmetro de
regularização $L$.

Inicialize $\theta$ e $\theta_0$ com vetores de zeros. Para cada atualização
realizada até o momento (de $1$ a $nT$, inclusive), defina a taxa de aprendizado
como:

$$\eta = \frac{1}{\sqrt{t}}$$

onde $t$ é o contador de updates (número de passos dados até o momento).

> [!WARNING]
> Assim como no Perceptron, certifique-se de chamar
> `get_order(feature_matrix.shape[0])` para obter os índices dos pontos de
> dados a cada iteração.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def pegasos(feature_matrix, labels, T, L):
    # Seu código aqui
```

# 5. Discussão dos Algoritmos

Uma vez concluída a implementação dos 3 algoritmos de aprendizado, você deve
verificar qualitativamente suas implementações. No arquivo
[main.py](sentiment_analysis/main.py), incluímos um bloco de código que você deve
descomentar. Este código carrega um conjunto de dados 2D de `toy_data.tsv` e
treina seus modelos usando $T = 10$ e $\lambda = 0.2$.

O script [main.py](sentiment_analysis/main.py) calculará $\theta$ e $\theta_0$
para cada um dos algoritmos de aprendizado que você escreveu. Em seguida, chamará
a função `plot_toy_data` para plotar o modelo e a fronteira de decisão resultantes.

---

## Gráficos

Para verificar seus gráficos, informe os valores de $\theta$ e $\theta_0$ obtidos
para os três algoritmos (com precisão de até 4 casas decimais). Por exemplo, se
$\theta = (1, 0.5)$, informe como `1, 0.5`.

- **Para o algoritmo Perceptron**:
  - $\theta = 3.9174, 4.164$
  - $\theta_0 = -8.0$
- **Para o algoritmo Perceptron Médio**:
  - $\theta = 3.4783, 3.6111$
  - $\theta_0 = -6.373$
- **Para o algoritmo Pegasos**:
  - $\theta = 0.7346, 0.6300$
  - $\theta_0 = -1.2195$

---

## Convergência

Como você implementou três algoritmos de aprendizado diferentes para classificadores
lineares, é interessante investigar qual algoritmo realmente convergiria.
Execute-os com um número maior de iterações $T$ para verificar se o algoritmo
converge visualmente no gráfico. Você também pode checar se o vetor de parâmetros
$\theta$ converge na primeira casa decimal.

**Pergunta**: Qual dos seguintes algoritmos irá convergir neste conjunto de dados?
(Selecione todas as opções que se aplicam):

- [ ] Algoritmo Perceptron
- [x] Algoritmo Perceptron Médio
- [x] Algoritmo Pegasos

# 6. Analisador de Avaliações de Produtos

Agora que você verificou a corretude de suas implementações, está pronto para
enfrentar a tarefa principal deste projeto: construir um classificador que
rotula avaliações como positivas ou negativas usando características baseadas em
texto e os classificadores lineares que você implementou na seção anterior!

---

## Os Dados

Os dados consistem em várias avaliações, cada uma das quais foi rotulada com
$-1$ ou $+1$, correspondendo a uma avaliação negativa ou positiva,
respectivamente. Os dados originais foram divididos em quatro arquivos:

- `reviews_train.tsv` (4000 exemplos)
- `reviews_val.tsv` (500 exemplos)
- `reviews_test.tsv` (500 exemplos)

Para ter uma ideia de como os dados se parecem, sugerimos primeiro abrir os
arquivos com um editor de texto, programa de planilha ou outro pacote de
software científico (como pandas).

---

## Traduzindo avaliações em vetores de características

Converteremos textos de avaliação em vetores de características usando uma
abordagem de **bag of words** (sacola de palavras). Começamos compilando todas as
palavras que aparecem em um conjunto de treinamento de avaliações em um
**dicionário**, produzindo assim uma lista de $d$ palavras únicas.

Podemos então transformar cada uma das avaliações em um vetor de características
de comprimento $d$, definindo a coordenada $i$-ésima do vetor de características
como $1$ se a $i$-ésima palavra no dicionário aparecer na avaliação, ou $0$ caso
contrário. Por exemplo, considere dois documentos simples "Mary loves apples"
e "Red apples". Neste caso, o dicionário é o conjunto $\{Mary, loves, apples,
red\}$, e os documentos são representados como $(1; 1; 1; 0)$ e $(0; 0; 1; 1)$.

Um modelo bag of words pode ser facilmente expandido para incluir frases de
comprimento $m$. Um modelo de **unigrama** (unigram) é o caso para o qual $m =
1$. No exemplo, o dicionário de unigramas seria $(Mary; loves; apples; red)$. No
caso de **bigrama** (bigram), $m = 2$, o dicionário seria $(Mary loves; loves
apples; Red apples)$, e as representações para cada amostra seriam $(1; 1; 0)$ e
$(0; 0; 1)$. Nesta seção, você usará apenas as características de palavras de
unigrama. Essas funções já estão implementadas para você na função bag of
words.

No arquivo [utils.py](sentiment_analysis/utils.py), fornecemos a você a função
`load_data`, que pode ser usada para ler os arquivos `.tsv` e retornar os rótulos
e textos. Também fornecemos a função `bag_of_words` no arquivo
[project1.py](sentiment_analysis/project1.py), que recebe os dados brutos e
retorna o dicionário de palavras unigramas. O dicionário resultante é uma
entrada para a função `extract_bow_feature_vectors` que você editará para
computar uma matriz de características de uns e zeros que pode ser usada como
entrada para os algoritmos de classificação. Usando a matriz de características e
sua implementação dos algoritmos de aprendizado anteriores, você será capaz de
computar $\theta$ e $\theta_0$.

# 7. Classificação e Acurácia

Agora precisamos de uma maneira de usar de fato nosso modelo para classificar os
pontos de dados. Nesta seção, você implementará uma maneira de classificar os
pontos de dados usando os parâmetros do seu modelo e, em seguida, medirá a
acurácia do seu modelo.

---

## Classificação

Implemente uma função de classificação que usa $\theta$ e $\theta_0$ para
classificar um conjunto de pontos de dados. São fornecidos a matriz de
características, $\theta$ e $\theta_0$. Esta função deve retornar um array numpy
de $-1$s e $1$s. Se uma previsão for **maior que zero**, ela deve ser
considerada uma classificação positiva.

> [!TIP]
> Como nos exercícios anteriores, quando $x$ é um float, "$x = 0$" deve ser
> verificado com $|x| < \epsilon$ para evitar imprecisões numéricas.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def classify(feature_matrix, theta, theta_0):
    # Seu código aqui
```

---

## Acurácia

A acurácia do classificador mede o quão bem ele prevê os rótulos corretos sobre
um dataset. Fornecemos a você a função `accuracy` para computar a proporção de
previsões corretas:

```python
def accuracy(preds, targets):
    return (preds == targets).mean()
```

Você deve usar esta função juntamente com as funções que implementou até agora
para implementar `classifier_accuracy` no arquivo
[project1.py](sentiment_analysis/project1.py). A função deve receber 6 argumentos:

- Uma função classificadora que, por si só, recebe argumentos
  `(feature_matrix, labels, **kwargs)`
- A matriz de características de treinamento
- A matriz de características de validação
- Os rótulos de treinamento
- Os rótulos de validação
- O argumento `**kwargs` a ser passado para a função classificadora

Esta função deve treinar o classificador fornecido usando os dados de treinamento
e, em seguida, computar a acurácia da classificação tanto nos dados de
treinamento quanto nos de validação. O retorno deve ser uma tupla onde o primeiro
valor é a acurácia de treinamento e o segundo é a acurácia de validação.

Implemente a seguinte função no arquivo [project1.py](sentiment_analysis/project1.py):

```python
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    # Seu código aqui
```

---

## Acurácia de Linha de Base (Baseline Accuracy)

Agora, realize os seguintes passos para obter o desempenho inicial:

1. Edite a função `extract_bow_feature_vectors` sob a verificação `if binarize:`
   no arquivo [project1.py](sentiment_analysis/project1.py) de modo que a matriz
   de características codifique a presença de uma palavra como $1$ se presente, e
   $0$ se não (ou seja, codificação binária/indicadora).
2. Descomente as linhas relevantes no arquivo
   [main.py](sentiment_analysis/main.py).
3. Execute o script [main.py](sentiment_analysis/main.py) e relate as acurácias
   de validação de cada algoritmo com $T = 10$ e $\lambda = 0.01$ (o valor de
   $\lambda$ se aplica apenas ao Pegasos).

> [!NOTE]
> Se você receber avisos sobre "Bag of words" ou "Extract bow feature vectors"
> ao executar o `test.py`, não se preocupe:
>
> - `WARN Bag of words : does not remove stopwords`
> - `WARN Extract bow feature vectors : uses binary indicators as features`

Informe os resultados obtidos nos campos abaixo:

- **Acurácia de validação do Perceptron**: `0.7160`
- **Acurácia de validação do Perceptron Médio**: `0.7980`
- **Acurácia de validação do Pegasos**: `0.7900`

# 8. Ajuste de Parâmetros

Você finalmente tem seus algoritmos funcionando e uma maneira de medir o
desempenho! Mas ainda não está claro quais valores os hiperparâmetros como $T$
e $\lambda$ devem ter. Nesta seção, você ajustará esses hiperparâmetros para
maximizar o desempenho de cada modelo.

Uma maneira de ajustar hiperparâmetros é realizar uma busca em grade (grid search)
sobre todas as combinações possíveis de valores. Por razões de eficiência,
ajustaremos um único parâmetro de cada vez, mantendo todos os outros constantes.

No arquivo [main.py](sentiment_analysis/main.py), descomente a seção do
Problema 8 para executar o algoritmo de ajuste fornecido pela equipe a partir de
[utils.py](sentiment_analysis/utils.py).

Para os propósitos desta tarefa, tente os seguintes valores:

- **Valores de $T$**: $[1, 5, 10, 15, 25, 50]$
- **Valores de $\lambda$**: $[0.001, 0.01, 0.1, 1, 10]$

Para o Pegasos, primeiro fixe $\lambda = 0.01$ para encontrar o melhor $T$ e,
em seguida, use o melhor $T$ encontrado para ajustar $\lambda$.

---

## Desempenho Após o Ajuste

Após rodar o ajuste no [main.py](sentiment_analysis/main.py), insira os melhores
valores encontrados e a acurácia de validação associada:

- **Algoritmo Perceptron**:
  - Melhor $T = 25$
  - Acurácia de validação = $0.7940$
- **Algoritmo Perceptron Médio**:
  - Melhor $T = 25$
  - Acurácia de validação = $0.8000$
- **Algoritmo Pegasos**:
  - Melhor $T = 25$
  - Melhor $\lambda = 0.01$
  - Acurácia de validação = $0.8060$

---

## Acurácia no conjunto de teste

Depois de escolher seu melhor classificador e seus hiperparâmetros correspondentes,
use-o para computar a acurácia de teste no conjunto de teste. No script
[main.py](sentiment_analysis/main.py), a matriz de características e os rótulos de
teste são fornecidos como `test_bow_features` e `test_labels`.

- **Acurácia no conjunto de teste**: `0.8020`

---

## Os unigramas mais explicativos

Podemos descobrir quais unigramas foram os mais impactantes na previsão de rótulos
**positivos** olhando para os maiores coeficientes positivos em $\theta$.
Descomente a parte correspondente no arquivo [main.py](sentiment_analysis/main.py)
para chamar `utils.most_explanatory_word`.

Relate os dez unigramas mais explicativos para classificação positiva abaixo:

1. Top 1: `delicious`
2. Top 2: `great`
3. Top 3: `!`
4. Top 4: `best`
5. Top 5: `perfect`
6. Top 6: `loves`
7. Top 7: `wonderful`
8. Top 8: `glad`
9. Top 9: `love`
10. Top 10: `quickly`

*Dica: Você também pode experimentar encontrar os unigramas que foram os mais
impactantes na previsão de rótulos negativos.*

# 9. Engenharia de Características

Frequentemente, a maneira como os dados são representados pode ter um impacto
significativo no desempenho de um método de aprendizado de máquina. Tente
melhorar o desempenho do seu melhor classificador usando características
diferentes. Neste problema, praticaremos duas variantes simples da representação
bag of words (BoW).

---

## Remover Stop Words

Tente implementar a remoção de stop words no seu código de engenharia de
características. Especificamente, carregue o arquivo
[stopwords.txt](sentiment_analysis/stopwords.txt), remova as palavras contidas no
arquivo do seu dicionário editando a função `bag_of_words` no arquivo
[project1.py](sentiment_analysis/project1.py), e use as características
construídas a partir do novo dicionário para treinar seu modelo e fazer
previsões.

Compare seu resultado nos dados de **teste** usando o algoritmo Pegasos com
$T = 25$ e $L = 0.01$ quando você remove as palavras do
[stopwords.txt](sentiment_analysis/stopwords.txt) do seu dicionário.

> [!TIP]
> Em vez de substituir a matriz de características com colunas de zeros para as
> stop words, você pode modificar a função `bag_of_words` para evitar que as
> stop words sejam adicionadas ao dicionário.

- **Acurácia no conjunto de teste usando o dicionário original**: $0.8020$
- **Acurácia no conjunto de teste usando o dicionário com stop words removidas**: `[____]`

---

## Alterar Características Binárias para Características de Contagem

Novamente, use o mesmo algoritmo de aprendizado e a mesma característica do
problema anterior. No entanto, quando você computar o vetor de características
de uma palavra, use a sua contagem de ocorrências em cada documento em vez de
um indicador binário (0 ou 1).

> [!TIP]
> Você é livre para modificar a função `extract_bow_feature_vectors` no arquivo
> [project1.py](sentiment_analysis/project1.py) para computar características de
> contagem.

- **Acurácia no conjunto de teste usando o dicionário com stop words removidas e características de contagem**: `[____]`

*Tente comparar seu resultado com o do problema anterior e veja a discussão da
solução após responder à pergunta.*

---

## Outras Explorações de Características

Algumas características adicionais que você pode querer explorar são:

- **Comprimento do texto** (número de caracteres ou de palavras);
- **Ocorrência de palavras totalmente em maiúsculas** (ex: "AMAZING", "DON'T BUY THIS");
- **Embeddings de palavras** (representações vetoriais densas).

Além de adicionar novas características, você também pode alterar o conjunto de
características unigramas original. Por exemplo:

- **Limiarizar a frequência mínima**: Estipular um número mínimo de vezes que
  uma palavra deve aparecer no conjunto de dados antes de adicioná-la ao dicionário.
  Por exemplo, palavras que ocorrem menos de três vezes em todo o conjunto de dados
  de treinamento poderiam ser consideradas irrelevantes e ser descartadas. Isso
  ajuda a reduzir o número de colunas e evitar overfitting.

Existem também muitas outras coisas que você pode alterar ao treinar seu modelo.
Tente qualquer abordagem que possa ajudá-lo a capturar melhor o sentimento das
avaliações. Vale a pena analisar o conjunto de dados e pensar em alternativas.
Lembre-se de que nem todas as características irão necessariamente melhorar a
acurácia, por isso é recomendado experimentar opções simples antes de tentar
modelos muito complexos.
