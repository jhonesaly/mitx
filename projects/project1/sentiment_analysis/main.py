import project1 as p1
import utils
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')


#-------------------------------------------------------------------------------
# Data loading.
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Carregando datasets de reviews...")
    train_data = utils.load_data('reviews_train.tsv')
    val_data = utils.load_data('reviews_val.tsv')
    test_data = utils.load_data('reviews_test.tsv')

    train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
    val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
    test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

    # Verifica se bag_of_words e extract_bow_feature_vectors já estão implementados
    bow_ready = False
    try:
        dictionary = p1.bag_of_words(train_texts)
        train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
        val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
        test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
        bow_ready = True
        print("Processamento de Bag of Words (BoW) carregado com sucesso!")
    except NotImplementedError:
        print("\nAviso: bag_of_words ou extract_bow_feature_vectors não implementados ainda.")
        print("Os problemas que dependem do texto (7, 8) continuarão comentados.")

    #-------------------------------------------------------------------------------
    # Problem 5: Discussão dos Algoritmos (Visualização do Toy Dataset)
    #-------------------------------------------------------------------------------
    print("\n--- Executando Problem 5 (Visualização do Toy Dataset) ---")
    toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

    T = 10
    L = 0.2

    thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
    thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
    thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

    def plot_toy_results(algo_name, thetas):
        print('theta for', algo_name, 'is', ', '.join(map(str, list(thetas[0]))))
        print('theta_0 for', algo_name, 'is', str(thetas[1]))
        
        # Customização estética dos gráficos para visualização premium
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        
        # Plota os pontos com cores vibrantes e marcadores distintos
        colors = ['dodgerblue' if label == 1 else 'darkorange' for label in toy_labels]
        markers = ['^' if label == 1 else 's' for label in toy_labels]
        
        for idx in range(len(toy_labels)):
            plt.scatter(toy_features[idx, 0], toy_features[idx, 1], 
                        c=colors[idx], marker=markers[idx], s=75, edgecolors='black', zorder=5)
                        
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.xlim(xmin - 0.5, xmax + 0.5)
        plt.ylim(ymin - 0.5, ymax + 0.5)
        xmin, xmax = plt.xlim()
        
        # Plota a fronteira de decisão
        theta, theta_0 = thetas
        xs = np.linspace(xmin, xmax, 100)
        ys = -(theta[0]*xs + theta_0) / (theta[1] + 1e-16)
        
        plt.plot(xs, ys, 'k-', linewidth=2.5, label='Fronteira de Decisao')
        
        # Títulos e labels
        algo_title = ' '.join((word.capitalize() for word in algo_name.split(' ')))
        plt.title(f'Fronteira de Decisao ({algo_title})', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('x1', fontsize=10)
        plt.ylabel('x2', fontsize=10)
        plt.grid(True, linestyle=':')
        plt.legend(loc='upper right')
        
        # Salva o arquivo PNG localmente
        filename = f"toy_{algo_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        # Copia a imagem para o diretório de artefatos para o usuário visualizar
        artifact_dir = r"C:\Users\jhone\.gemini\antigravity-ide\brain\1a9fd251-be8f-46f8-94bc-ed648e779640"
        if os.path.exists(artifact_dir):
            try:
                import shutil
                shutil.copy(filename, os.path.join(artifact_dir, filename))
            except Exception as e:
                pass
                
        plt.show()

    plot_toy_results('Perceptron', thetas_perceptron)
    plot_toy_results('Average Perceptron', thetas_avg_perceptron)
    plot_toy_results('Pegasos', thetas_pegasos)

    #-------------------------------------------------------------------------------
    # Problem 7 (Opcional - Ativar quando p1.classifier_accuracy estiver implementado)
    #-------------------------------------------------------------------------------
    if bow_ready:
        print("\n--- Executando Problem 7 ---")
        T = 10
        L = 0.01
        
        pct_train_accuracy, pct_val_accuracy = \
           p1.classifier_accuracy(p1.perceptron, train_bow_features, val_bow_features, train_labels, val_labels, T=T)
        print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
        print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
        
        avg_pct_train_accuracy, avg_pct_val_accuracy = \
           p1.classifier_accuracy(p1.average_perceptron, train_bow_features, val_bow_features, train_labels, val_labels, T=T)
        print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
        print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
        
        avg_peg_train_accuracy, avg_peg_val_accuracy = \
           p1.classifier_accuracy(p1.pegasos, train_bow_features, val_bow_features, train_labels, val_labels, T=T, L=L)
        print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
        print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

    #-------------------------------------------------------------------------------
    # Problem 8 (Opcional - Ativar quando tuning estiver pronto)
    #-------------------------------------------------------------------------------
    # if bow_ready:
    #     print("\n--- Executando Problem 8 (Tuning) ---")
    #     data = (train_bow_features, train_labels, val_bow_features, val_labels)
    #     Ts = [1, 5, 10, 15, 25, 50]
    #     Ls = [0.001, 0.01, 0.1, 1, 10]
    #     
    #     pct_tune_results = utils.tune_perceptron(Ts, *data)
    #     print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
    #     print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))
    #     
    #     avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
    #     print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
    #     print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))
    #     
    #     fix_L = 0.01
    #     peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
    #     print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
    #     print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))
    #     
    #     fix_T = Ts[np.argmax(peg_tune_results_T[1])]
    #     peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
    #     print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
    #     print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))
    #     
    #     utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
    #     utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
    #     utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
    #     utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)
