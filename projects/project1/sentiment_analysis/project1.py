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
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    # Your code here
    raise NotImplementedError



def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    # Your code here
    raise NotImplementedError
    for t in range(T):
        for i in get_order(nsamples):
            # Your code here
            raise NotImplementedError
    # Your code here
    raise NotImplementedError



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    # Your code here
    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    # Your code here
    raise NotImplementedError



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    # Your code here
    raise NotImplementedError



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
    
    print("Gerando visualizações para hinge_loss_single e hinge_loss_full...")
    
    # ─── Exemplo 1: Visualização da Perda Hinge vs Margem Funcional ───────────
    margins = np.linspace(-2.0, 3.0, 100)
    # Hinge loss para cada margem: max(0, 1 - m)
    losses = np.maximum(0.0, 1.0 - margins)
    
    plt.figure(figsize=(12, 5.5))
    
    # Subplot 1: Função de Perda Hinge
    plt.subplot(1, 2, 1)
    plt.plot(margins, losses, label="Hinge Loss", color="red", linewidth=2.5)
    plt.axvline(x=1.0, color="gray", linestyle="--", label="Margem de Segurança (m=1)")
    plt.axvline(x=0.0, color="black", linestyle="-", label="Fronteira de Decisão (m=0)")
    plt.title("Hinge Loss vs Margem Funcional")
    plt.xlabel("Margem Funcional (y * (θ·x + θ₀))")
    plt.ylabel("Perda Hinge (L)")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    # ─── Exemplo 2: Perda Hinge Média em um Dataset Toy 2D ───────────────────
    # Criando 4 pontos em 2D
    # X1 = [1, 2] (y = 1) -> Classificado correto, fora da margem
    # X2 = [1, 0.5] (y = 1) -> Classificado correto, mas dentro da margem
    # X3 = [-1, -1] (y = -1) -> Classificado correto, fora da margem
    # X4 = [0.5, -0.5] (y = -1) -> Classificado incorreto (do lado positivo)
    toy_features = np.array([
        [1.0, 2.0],
        [1.0, 0.5],
        [-1.0, -1.0],
        [0.5, -0.5]
    ])
    toy_labels = np.array([1, 1, -1, -1])
    
    # Classificador: θ = [1, 1], θ_0 = 0
    theta = np.array([1.0, 1.0])
    theta_0 = 0.0
    
    # Computando perdas individuais e margens usando nossas funções
    margins_toy = functional_margin(toy_features, toy_labels, theta, theta_0)
    losses_toy = np.maximum(0.0, 1.0 - margins_toy)
    avg_loss = hinge_loss_full(toy_features, toy_labels, theta, theta_0)
    
    # Subplot 2: Dataset e Fronteira
    plt.subplot(1, 2, 2)
    # Plota a fronteira de decisão x2 = -x1 (já que θ1*x1 + θ2*x2 = 0 => x2 = -x1)
    x1_vals = np.linspace(-2.0, 2.0, 100)
    x2_vals = -x1_vals
    plt.plot(x1_vals, x2_vals, color="black", label="Fronteira (θ·x = 0)")
    
    # Plota as linhas de margem de segurança (θ·x = +1 e θ·x = -1)
    plt.plot(x1_vals, x2_vals + 1/theta[1], color="gray", linestyle=":", label="Margem de Segurança")
    plt.plot(x1_vals, x2_vals - 1/theta[1], color="gray", linestyle=":")
    
    # Plota os pontos
    for i in range(len(toy_labels)):
        color = "blue" if toy_labels[i] == 1 else "orange"
        marker = "^" if toy_labels[i] == 1 else "s"
        plt.scatter(toy_features[i, 0], toy_features[i, 1], color=color, marker=marker, s=120, edgecolors='black', zorder=5)
        plt.text(toy_features[i, 0] + 0.1, toy_features[i, 1], f"Perda: {losses_toy[i]:.2f}", fontsize=10, weight='bold')
        
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 3.0)
    plt.title(f"Hinge Loss Média no Dataset: {avg_loss:.3f}")
    plt.xlabel("Característica 1 (x1)")
    plt.ylabel("Característica 2 (x2)")
    plt.grid(True, linestyle=":")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("hinge_loss_visualization.png")
    print("Visualização salva em 'hinge_loss_visualization.png' com sucesso!")