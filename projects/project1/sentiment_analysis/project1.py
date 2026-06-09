from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def functional_margin(features, labels, theta, theta_0):
    return labels * (np.dot(features, theta) + theta_0)



def hinge_loss_single(feature_vector, label, theta, theta_0):
    return max(0.0, 1.0 - functional_margin(feature_vector, label, theta, theta_0))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    margins = functional_margin(feature_matrix, labels, theta, theta_0)
    losses = np.maximum(0.0, 1.0 - margins)
    return float(np.mean(losses))




def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    margin = functional_margin(feature_vector, label, current_theta, current_theta_0)
    if margin <= 1e-9:
        return current_theta + label * feature_vector, current_theta_0 + label
    else:
        return current_theta, current_theta_0


def initialize_parameters(n_features):
    return np.zeros(n_features), 0.0



def perceptron(feature_matrix, labels, T):
    nsamples, nfeatures = feature_matrix.shape
    theta, theta_0 = initialize_parameters(nfeatures)
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )
    return theta, theta_0



def average_perceptron(feature_matrix, labels, T):
    nsamples, nfeatures = feature_matrix.shape
    theta, theta_0 = initialize_parameters(nfeatures)
    theta_sum, theta_0_sum = initialize_parameters(nfeatures)
    counter = 0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )
            theta_sum += theta
            theta_0_sum += theta_0
            counter += 1
    return theta_sum / counter, theta_0_sum / counter


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    margin = functional_margin(feature_vector, label, theta, theta_0)
    if margin <= 1.0 + 1e-9:
        theta = (1.0 - eta * L) * theta + eta * label * feature_vector
        theta_0 = theta_0 + eta * label
    else:
        theta = (1.0 - eta * L) * theta
    return theta, theta_0



def pegasos(feature_matrix, labels, T, L):
    nsamples, nfeatures = feature_matrix.shape
    theta, theta_0 = initialize_parameters(nfeatures)
    t_counter = 0
    for t in range(T):
        for i in get_order(nsamples):
            t_counter += 1
            eta = 1.0 / np.sqrt(t_counter)
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0
            )
    return theta, theta_0



#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    # Your code here
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Your code here
    raise NotImplementedError



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
    raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here
    raise NotImplementedError
    
    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        # Your code here
        raise NotImplementedError
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # Determina o diretório deste script para salvar as imagens na pasta correta
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Novo dataset toy expandido contendo mais pontos e 2 outliers
    # y = 1 (Azul/Triângulo), y = -1 (Laranja/Quadrado)
    toy_features = np.array([
        [1.0, 2.0],     # Triângulo Azul
        [2.0, 1.0],     # Triângulo Azul
        [1.5, 1.5],     # Triângulo Azul
        [2.5, 2.0],     # Triângulo Azul
        [2.0, 2.5],     # Triângulo Azul
        [0.0, 0.0],     # Quadrado Laranja
        [-1.0, 0.5],    # Quadrado Laranja
        [-0.5, -0.5],   # Quadrado Laranja
        [-1.5, 0.0],    # Quadrado Laranja
        [-1.0, -1.0],   # Quadrado Laranja
        # Outliers:
        [1.5, 2.2],     # Quadrado Laranja (infiltrado no cluster azul)
        [-0.8, -0.2]    # Triângulo Azul (infiltrado no cluster laranja)
    ])
    toy_labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1])

    # Funções de treinamento sequenciais e determinísticas para os plots
    def run_perceptron_sequential(feature_matrix, labels, T):
        nsamples, nfeatures = feature_matrix.shape
        theta, theta_0 = initialize_parameters(nfeatures)
        for t in range(T):
            for i in range(nsamples):
                theta, theta_0 = perceptron_single_step_update(
                    feature_matrix[i], labels[i], theta, theta_0
                )
        return theta, theta_0

    def run_average_perceptron_sequential(feature_matrix, labels, T):
        nsamples, nfeatures = feature_matrix.shape
        theta, theta_0 = initialize_parameters(nfeatures)
        theta_sum, theta_0_sum = initialize_parameters(nfeatures)
        counter = 0
        for t in range(T):
            for i in range(nsamples):
                theta, theta_0 = perceptron_single_step_update(
                    feature_matrix[i], labels[i], theta, theta_0
                )
                theta_sum += theta
                theta_0_sum += theta_0
                counter += 1
        return theta_sum / counter, theta_0_sum / counter

    def run_pegasos_sequential(feature_matrix, labels, T, L):
        nsamples, nfeatures = feature_matrix.shape
        theta, theta_0 = initialize_parameters(nfeatures)
        t_counter = 0
        for t in range(T):
            for i in range(nsamples):
                t_counter += 1
                eta = 1.0 / np.sqrt(t_counter)
                theta, theta_0 = pegasos_single_step_update(
                    feature_matrix[i], labels[i], L, eta, theta, theta_0
                )
        return theta, theta_0
    
    # ─── Visualização 1: Hinge Loss ──────────────────────────────────────────
    print("Gerando visualizações para hinge_loss_single e hinge_loss_full...")
    
    # Eixo de Margens para a curva Hinge
    margins = np.linspace(-3.0, 3.0, 100)
    losses = np.maximum(0.0, 1.0 - margins)
    
    plt.figure(figsize=(12, 5.5))
    
    # Subplot 1: Curva da Hinge Loss
    plt.subplot(1, 2, 1)
    plt.plot(margins, losses, label="Hinge Loss", color="red", linewidth=2.5)
    plt.axvline(x=1.0, color="gray", linestyle="--", label="Margem de Segurança (m=1)")
    plt.axvline(x=0.0, color="black", linestyle="-", label="Fronteira de Decisão (m=0)")
    plt.title("Hinge Loss vs Margem Funcional")
    plt.xlabel("Margem Funcional (y * (θ·x + θ₀))")
    plt.ylabel("Perda Hinge (L)")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: Dataset Toy e Margem de Segurança
    # Usando o classificador θ = [1.0, 1.0], θ_0 = 0.0
    theta_toy = np.array([1.0, 1.0])
    theta_0_toy = 0.0
    
    margins_toy = functional_margin(toy_features, toy_labels, theta_toy, theta_0_toy)
    losses_toy = np.maximum(0.0, 1.0 - margins_toy)
    avg_loss = hinge_loss_full(toy_features, toy_labels, theta_toy, theta_0_toy)
    
    plt.subplot(1, 2, 2)
    x1_vals = np.linspace(-3.0, 3.0, 100)
    x2_vals = -x1_vals
    plt.plot(x1_vals, x2_vals, color="black", label="Fronteira (θ·x = 0)")
    plt.plot(x1_vals, x2_vals + 1.0, color="gray", linestyle=":", label="Margem de Segurança")
    plt.plot(x1_vals, x2_vals - 1.0, color="gray", linestyle=":")
    
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        # Exibe a perda de cada ponto no gráfico
        plt.text(toy_features[i, 0] + 0.15, toy_features[i, 1], f"{losses_toy[i]:.1f}", fontsize=8, weight='bold')
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title(f"Hinge Loss Média no Dataset: {avg_loss:.3f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "hinge_loss_visualization.png"))
    plt.close()
    print("Visualização salva em 'hinge_loss_visualization.png' com sucesso!")
    
    # ─── Visualização 2: Perceptron - Evolução do Ajuste ─────────────────────
    print("Gerando visualizações para o Perceptron...")
    
    # Executa a primeira atualização sequencial (Passo 1)
    theta_zeros = np.zeros(2)
    theta_0_zeros = 0.0
    theta_step1, theta_0_step1 = perceptron_single_step_update(toy_features[0], toy_labels[0], theta_zeros, theta_0_zeros)
    
    # Executa a fronteira do perceptron padrão final após T = 5 épocas
    theta_final_std, theta_0_final_std = run_perceptron_sequential(toy_features, toy_labels, 5)
    
    plt.figure(figsize=(12, 5.5))
    
    # Subplot 1: Reta após o Passo 1 de atualização
    plt.subplot(1, 2, 1)
    # Reta após 1º update: x1 + 2x2 + 1 = 0 => x2 = -0.5*x1 - 0.5
    plt.plot(x1_vals, -0.5 * x1_vals - 0.5, color="green", linewidth=2.5, label="Fronteira após Passo 1")
    
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Perceptron: Após Passo 1 (θ = [1, 2], θ₀ = 1.0)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: Reta final após T=5 épocas (não converge perfeitamente devido aos outliers)
    plt.subplot(1, 2, 2)
    plt.plot(x1_vals, - (theta_final_std[0] * x1_vals + theta_0_final_std) / theta_final_std[1], 
             color="black", linewidth=3.0, label=f"Fronteira Final: θ={theta_final_std}, θ₀={theta_0_final_std:.1f}")
    
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Perceptron Padrão: Estado Final (T = 5 Épocas)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "perceptron_visualization.png"))
    plt.close()
    print("Visualização do Perceptron salva em 'perceptron_visualization.png' com sucesso!")
 
    # ─── Visualização 3: Caminho de Otimização (Clássico vs Average) ──────────
    print("Gerando visualização do caminho do Perceptron (reta se ajeitando)...")
    
    # Rastreia a evolução dos pesos clássicos e das médias corrente a cada update
    theta_path = np.zeros(2)
    theta_0_path = 0.0
    theta_sum = np.zeros(2)
    theta_0_sum = 0.0
    counter = 0
    
    history_std = []
    history_avg = []
    
    for epoch in range(5):
        for i in range(len(toy_labels)):
            x_i = toy_features[i]
            y_i = toy_labels[i]
            updated = False
            if y_i * (np.dot(theta_path, x_i) + theta_0_path) <= 1e-9:
                theta_path = theta_path + y_i * x_i
                theta_0_path = theta_0_path + y_i
                updated = True
            
            theta_sum += theta_path
            theta_0_sum += theta_0_path
            counter += 1
            
            if updated:
                history_std.append((theta_path.copy(), theta_0_path, f"Passo {len(history_std) + 1}"))
                history_avg.append((theta_sum.copy() / counter, theta_0_sum / counter, f"Média {len(history_avg) + 1}"))

    # Configuração da figura com 2 subplots lado a lado para comparação direta
    plt.figure(figsize=(15, 7))
    x1_line_path = np.linspace(-3.0, 3.0, 100)
    
    # Cores bem distintas e contrastantes para as retas intermediárias
    colors_list = ["green", "blue", "purple", "orange", "magenta", "cyan"]
    
    # Limita o número de passos intermediários mostrados para não poluir
    max_steps_to_plot = 5
    plot_indices = list(range(min(len(history_std), max_steps_to_plot)))
    if len(history_std) > max_steps_to_plot and (len(history_std) - 1) not in plot_indices:
        plot_indices.append(len(history_std) - 1)
        
    # --- Subplot 1: Perceptron Clássico ---
    plt.subplot(1, 2, 1)
    # Desenha o Passo 0 ilustrativo
    plt.plot(x1_line_path, -x1_line_path, color="red", linestyle=":", 
             label="Passo 0 (Inicial): θ = [0, 0], θ₀ = 0.0 (Ilustrativa)", linewidth=2)
             
    for idx, k in enumerate(plot_indices):
        w, b, label_name = history_std[k]
        color_val = colors_list[idx % len(colors_list)]
        if w[1] == 0:
            plt.axvline(x=-b/w[0], color=color_val, linestyle="--", label=f"{label_name}: θ={w}, θ₀={b:.1f}", linewidth=2)
        else:
            plt.plot(x1_line_path, - (w[0] * x1_line_path + b) / w[1], 
                     color=color_val, linestyle="--", label=f"{label_name}: θ={w}, θ₀={b:.1f}", linewidth=2)
                     
    # Destaca o separador final em linha preta contínua mais grossa
    w_f, b_f, _ = history_std[-1]
    plt.plot(x1_line_path, - (w_f[0] * x1_line_path + b_f) / w_f[1], 
             color="black", label=f"Final (Passo {len(history_std)}): θ={w_f}, θ₀={b_f:.1f}", linewidth=3)
             
    # Plota os pontos
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Caminho do Perceptron Clássico (Altamente Instável)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend(loc="upper right", fontsize=8)
    
    # --- Subplot 2: Average Perceptron (Média Acumulada) ---
    plt.subplot(1, 2, 2)
    # Desenha o Passo 0 ilustrativo para a média
    plt.plot(x1_line_path, -x1_line_path, color="red", linestyle=":", 
             label="Média Inicial: θ = [0, 0], θ₀ = 0.0 (Ilustrativa)", linewidth=2)
             
    for idx, k in enumerate(plot_indices):
        w, b, label_name = history_avg[k]
        color_val = colors_list[idx % len(colors_list)]
        w_str = f"[{w[0]:.2f} {w[1]:.2f}]"
        if w[1] == 0:
            plt.axvline(x=-b/w[0], color=color_val, linestyle="--", label=f"{label_name}: θ={w_str}, θ₀={b:.2f}", linewidth=2)
        else:
            plt.plot(x1_line_path, - (w[0] * x1_line_path + b) / w[1], 
                     color=color_val, linestyle="--", label=f"{label_name}: θ={w_str}, θ₀={b:.2f}", linewidth=2)
                     
    # Destaca a média final em linha preta contínua mais grossa
    w_f_avg, b_f_avg, _ = history_avg[-1]
    w_f_str = f"[{w_f_avg[0]:.2f} {w_f_avg[1]:.2f}]"
    plt.plot(x1_line_path, - (w_f_avg[0] * x1_line_path + b_f_avg) / w_f_avg[1], 
             color="black", label=f"Média Final: θ={w_f_str}, θ₀={b_f_avg:.2f}", linewidth=3)
             
    # Plota os pontos
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Caminho do Average Perceptron (Estabilizando no Tempo)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "perceptron_path_visualization.png"))
    plt.close()
    print("Visualização do caminho comparativo salva em 'perceptron_path_visualization.png' com sucesso!")

    # ─── Visualização 4: Average Perceptron vs Perceptron Padrão ──────────────
    print("Gerando visualizações para o Average Perceptron...")
    
    # Execuções sequenciais para T = 1 e T = 5
    theta_std_1, theta_0_std_1 = run_perceptron_sequential(toy_features, toy_labels, 1)
    theta_avg_1, theta_0_avg_1 = run_average_perceptron_sequential(toy_features, toy_labels, 1)
    
    theta_std_5, theta_0_std_5 = run_perceptron_sequential(toy_features, toy_labels, 5)
    theta_avg_5, theta_0_avg_5 = run_average_perceptron_sequential(toy_features, toy_labels, 5)
    
    plt.figure(figsize=(12, 5.5))
    x1_line = np.linspace(-3.0, 3.0, 100)
    
    # Subplot 1: T = 1 época
    plt.subplot(1, 2, 1)
    plt.plot(x1_line, - (theta_std_1[0] * x1_line + theta_0_std_1) / theta_std_1[1],
             color="black", linewidth=2.5, label="Perceptron Padrão")
    plt.plot(x1_line, - (theta_avg_1[0] * x1_line + theta_0_avg_1) / theta_avg_1[1],
             color="purple", linestyle="--", linewidth=2.5, label="Average Perceptron")
             
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Average vs Padrão (T = 1 Época)\nNote a oscilação inicial")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: T = 5 épocas (Demonstração de robustez a outliers)
    plt.subplot(1, 2, 2)
    plt.plot(x1_line, - (theta_std_5[0] * x1_line + theta_0_std_5) / theta_std_5[1],
             color="black", linewidth=2.5, label="Perceptron Padrão")
    plt.plot(x1_line, - (theta_avg_5[0] * x1_line + theta_0_avg_5) / theta_avg_5[1],
             color="purple", linestyle="--", linewidth=2.5, label="Average Perceptron")
             
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Average vs Padrão (T = 5 Épocas)\nAverage é muito mais estável")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "average_perceptron_visualization.png"))
    plt.close()
    print("Visualização do Average Perceptron salva em 'average_perceptron_visualization.png' com sucesso!")

    # ─── Visualização 5: Pegasos (Impacto da Regularização L) ───────────────
    print("Gerando visualizações para o Pegasos...")
    
    # Execuções do Pegasos com L=0.01 (fraca) e L=0.5 (forte)
    theta_peg_weak, theta_0_peg_weak = run_pegasos_sequential(toy_features, toy_labels, 5, 0.01)
    theta_peg_strong, theta_0_peg_strong = run_pegasos_sequential(toy_features, toy_labels, 5, 0.5)
    
    # Reta de referência do Perceptron Padrão e Average Perceptron
    theta_std, theta_0_std = run_perceptron_sequential(toy_features, toy_labels, 5)
    theta_avg, theta_0_avg = run_average_perceptron_sequential(toy_features, toy_labels, 5)
    
    plt.figure(figsize=(12, 5.5))
    x1_line = np.linspace(-3.0, 3.0, 100)
    
    # Subplot 1: Regularização Fraca (L = 0.01)
    plt.subplot(1, 2, 1)
    plt.plot(x1_line, - (theta_peg_weak[0] * x1_line + theta_0_peg_weak) / theta_peg_weak[1],
             color="blue", linewidth=2.5, label="Pegasos (L=0.01)")
    plt.plot(x1_line, - (theta_std[0] * x1_line + theta_0_std) / theta_std[1],
             color="black", linestyle=":", linewidth=2, label="Perceptron Padrão")
    plt.plot(x1_line, - (theta_avg[0] * x1_line + theta_0_avg) / theta_avg[1],
             color="purple", linestyle="--", linewidth=2.0, label="Average Perceptron")
             
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Pegasos: Regularização Fraca (L = 0.01)\nReta é puxada pelos outliers")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: Regularização Forte (L = 0.5)
    plt.subplot(1, 2, 2)
    plt.plot(x1_line, - (theta_peg_strong[0] * x1_line + theta_0_peg_strong) / theta_peg_strong[1],
             color="blue", linewidth=2.5, label="Pegasos (L=0.5)")
    plt.plot(x1_line, - (theta_std[0] * x1_line + theta_0_std) / theta_std[1],
             color="black", linestyle=":", linewidth=2, label="Perceptron Padrão")
    plt.plot(x1_line, - (theta_avg[0] * x1_line + theta_0_avg) / theta_avg[1],
             color="purple", linestyle="--", linewidth=2.0, label="Average Perceptron")
             
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title("Pegasos: Regularização Forte (L = 0.5)\nIgnora outliers e foca na Margem Máxima")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "pegasos_visualization.png"))
    plt.close()
    print("Visualização do Pegasos salva em 'pegasos_visualization.png' com sucesso!")

    # ─── Visualização 6: Otimização do Pegasos (Concavidade e Descida do Custo) 
    print("Gerando visualizações para a concavidade e otimização do Pegasos...")
    
    L_val = 0.5
    theta_path = np.zeros(2)
    theta_0_path = 0.0
    t_counter = 0
    
    peg_theta_history = [theta_path.copy()]
    peg_cost_history = []
    
    def compute_svm_cost(features, labels, w, w0, L):
        reg = 0.5 * L * np.sum(w ** 2)
        h_loss = hinge_loss_full(features, labels, w, w0)
        return reg + h_loss
        
    initial_cost = compute_svm_cost(toy_features, toy_labels, theta_path, theta_0_path, L_val)
    peg_cost_history.append(initial_cost)
    
    for t in range(5):
        for i in range(len(toy_labels)):
            t_counter += 1
            eta = 1.0 / np.sqrt(t_counter)
            theta_path, theta_0_path = pegasos_single_step_update(
                toy_features[i], toy_labels[i], L_val, eta, theta_path, theta_0_path
            )
            peg_theta_history.append(theta_path.copy())
            peg_cost_history.append(compute_svm_cost(toy_features, toy_labels, theta_path, theta_0_path, L_val))
            
    peg_theta_history = np.array(peg_theta_history)
    peg_cost_history = np.array(peg_cost_history)
    
    plt.figure(figsize=(12, 5.5))
    
    # Subplot 1: Curva de Otimização (Queda do Custo no Tempo)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(peg_cost_history)), peg_cost_history, color="blue", linewidth=2.5, marker="o", markersize=4, label="Custo Objetivo J(θ, θ₀)")
    plt.title("Otimização: Queda do Custo Objetivo J(θ, θ₀)\nao longo das iterações")
    plt.xlabel("Número de Updates (t)")
    plt.ylabel("Custo Objetivo J(θ, θ₀)")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: Superfície de Custo (Contorno) e Trajetória
    plt.subplot(1, 2, 2)
    
    theta_opt = peg_theta_history[-1]
    theta1_grid = np.linspace(theta_opt[0] - 2.0, theta_opt[0] + 2.0, 100)
    theta2_grid = np.linspace(theta_opt[1] - 2.0, theta_opt[1] + 2.0, 100)
    
    T1, T2 = np.meshgrid(theta1_grid, theta2_grid)
    Z = np.zeros_like(T1)
    
    opt_theta_0 = theta_0_path
    
    for row in range(len(theta2_grid)):
        for col in range(len(theta1_grid)):
            w_grid = np.array([T1[row, col], T2[row, col]])
            Z[row, col] = compute_svm_cost(toy_features, toy_labels, w_grid, opt_theta_0, L_val)
            
    contour = plt.contourf(T1, T2, Z, levels=25, cmap="viridis")
    plt.colorbar(contour, label="Custo Objetivo J")
    
    plt.plot(peg_theta_history[:, 0], peg_theta_history[:, 1], color="red", linestyle="-", marker="x", markersize=6, label="Trajetória de θ", linewidth=1.5)
    
    plt.scatter(0.0, 0.0, color="yellow", marker="*", s=200, edgecolors="black", zorder=6, label="Início: θ=[0,0]")
    plt.scatter(theta_opt[0], theta_opt[1], color="red", marker="P", s=150, edgecolors="white", zorder=6, label=f"Mínimo: θ*=[{theta_opt[0]:.2f}, {theta_opt[1]:.2f}]")
    
    plt.title("Concavidade: Contorno da Função de Custo Convexa\ncom a trajetória de descida do Pegasos")
    plt.xlabel("Parâmetro θ₁")
    plt.ylabel("Parâmetro θ₂")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "pegasos_objective_optimization.png"))
    plt.close()
    print("Visualização da otimização do Pegasos salva em 'pegasos_objective_optimization.png' com sucesso!")

    # ─── Visualização 7: Impacto do Hiperparâmetro L ─────────────────────────
    print("Gerando visualizações para o impacto do hiperparâmetro L...")
    
    L_values = np.logspace(-4, 1, 100)
    theta_norms = []
    hinge_losses = []
    total_costs = []
    
    for L_val in L_values:
        w, w0 = run_pegasos_sequential(toy_features, toy_labels, T=5, L=L_val)
        w_norm = np.linalg.norm(w)
        h_loss = hinge_loss_full(toy_features, toy_labels, w, w0)
        obj_cost = 0.5 * L_val * (w_norm ** 2) + h_loss
        
        theta_norms.append(w_norm)
        hinge_losses.append(h_loss)
        total_costs.append(obj_cost)
        
    plt.figure(figsize=(12, 5.5))
    
    # Subplot 1: Norma L2 de theta vs L
    plt.subplot(1, 2, 1)
    plt.plot(L_values, theta_norms, color="teal", linewidth=2.5, label="Norma L2 ||theta||")
    plt.xscale("log")
    plt.title("Impacto de L na Norma de theta")
    plt.xlabel("Parametro de Regularizacao L (Escala Log)")
    plt.ylabel("Norma L2 ||theta||")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # Subplot 2: Perda Hinge e Custo Total vs L
    plt.subplot(1, 2, 2)
    plt.plot(L_values, hinge_losses, color="crimson", linewidth=2.5, label="Perda Hinge Media")
    plt.plot(L_values, total_costs, color="darkblue", linestyle="--", linewidth=2.0, label="Custo Objetivo Total")
    plt.xscale("log")
    plt.title("Impacto de L nas Perdas e Custo")
    plt.xlabel("Parametro de Regularizacao L (Escala Log)")
    plt.ylabel("Valor da Perda / Custo")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "pegasos_L_impact_visualization.png"))
    plt.close()
    print("Visualização do impacto do L salva em 'pegasos_L_impact_visualization.png' com sucesso!")