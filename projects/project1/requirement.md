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
