import project1 as p1
import utils
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

ARTIFACT_DIR = r"C:\Users\jhone\.gemini\antigravity-ide\brain\4ed95c1d-c199-4037-8f2d-20147be8553b"



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
        os.makedirs('plots', exist_ok=True)
        filename = f"toy_{algo_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gráfico do toy dataset salvo em '{filepath}'")
        
        # Copia a imagem para o diretório de artefatos para o usuário visualizar
        if os.path.exists(ARTIFACT_DIR):
            try:
                import shutil
                shutil.copy(filepath, os.path.join(ARTIFACT_DIR, filename))
            except Exception:
                pass

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
    if bow_ready:
        print("\n--- Executando Problem 8 (Tuning) ---")
        data = (train_bow_features, train_labels, val_bow_features, val_labels)
        Ts = [1, 5, 10, 15, 25, 50]
        Ls = [0.001, 0.01, 0.1, 1, 10]
        
        def custom_plot_tune_results(algo_name, param_name, param_vals, acc_train, acc_val, filename):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 5.5))
            
            # Plota acurácias com estilo premium
            plt.plot(param_vals, acc_train, marker='o', markersize=8, color='royalblue', linewidth=2.5, label='Acurácia de Treino')
            plt.plot(param_vals, acc_val, marker='s', markersize=8, color='darkorange', linewidth=2.5, label='Acurácia de Validação')
            
            # Estilização
            algo_title = ' '.join((word.capitalize() for word in algo_name.split(' ')))
            param_title = param_name.capitalize()
            plt.title(f'Acurácia vs {param_title}\n({algo_title})', fontsize=12, fontweight='bold', pad=15)
            plt.xlabel(param_title, fontsize=10, labelpad=8)
            plt.ylabel('Acurácia', fontsize=10, labelpad=8)
            
            if param_name.upper() == 'L' or param_name.upper() == 'LAMBDA':
                plt.xscale('log')
                
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='lower right', frameon=True, facecolor='whitesmoke', edgecolor='gray')
            plt.tight_layout()
            
            # Salva o arquivo na pasta de plots
            os.makedirs('plots', exist_ok=True)
            filepath = os.path.join('plots', filename)
            plt.savefig(filepath, dpi=150)
            plt.close()
            print(f"Gráfico de tuning salvo em '{filepath}'")
            
            # Copia para a pasta de artefatos da conversa atual
            if os.path.exists(ARTIFACT_DIR):
                try:
                    import shutil
                    shutil.copy(filepath, os.path.join(ARTIFACT_DIR, filename))
                except Exception:
                    pass

        pct_tune_results = utils.tune_perceptron(Ts, *data)
        # Tabela Perceptron
        print("\n" + "="*50)
        print(" RESULTADOS DO TUNING: PERCEPTRON (T)")
        print("="*50)
        print("|    T    | Acurácia Treino | Acurácia Validação |")
        print("+---------+-----------------+--------------------+")
        for t, t_acc, v_acc in zip(Ts, pct_tune_results[0], pct_tune_results[1]):
            print(f"|  {t:5d}  |     {t_acc:7.2%}     |      {v_acc:7.2%}      |")
        print("="*50)
        print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))
        
        avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
        # Tabela Average Perceptron
        print("\n" + "="*50)
        print(" RESULTADOS DO TUNING: AVERAGE PERCEPTRON (T)")
        print("="*50)
        print("|    T    | Acurácia Treino | Acurácia Validação |")
        print("+---------+-----------------+--------------------+")
        for t, t_acc, v_acc in zip(Ts, avg_pct_tune_results[0], avg_pct_tune_results[1]):
            print(f"|  {t:5d}  |     {t_acc:7.2%}     |      {v_acc:7.2%}      |")
        print("="*50)
        print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))
        
        fix_L = 0.01
        peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
        # Tabela Pegasos T
        print("\n" + "="*50)
        print(" RESULTADOS DO TUNING: PEGASOS (T) [L=0.01]")
        print("="*50)
        print("|    T    | Acurácia Treino | Acurácia Validação |")
        print("+---------+-----------------+--------------------+")
        for t, t_acc, v_acc in zip(Ts, peg_tune_results_T[0], peg_tune_results_T[1]):
            print(f"|  {t:5d}  |     {t_acc:7.2%}     |      {v_acc:7.2%}      |")
        print("="*50)
        print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))
        
        fix_T = Ts[np.argmax(peg_tune_results_T[1])]
        peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
        # Tabela Pegasos L
        print("\n" + "="*50)
        print(f" RESULTADOS DO TUNING: PEGASOS (L) [T={fix_T}]")
        print("="*50)
        print("|    L    | Acurácia Treino | Acurácia Validação |")
        print("+---------+-----------------+--------------------+")
        for l, t_acc, v_acc in zip(Ls, peg_tune_results_L[0], peg_tune_results_L[1]):
            print(f"| {l:7.3f} |     {t_acc:7.2%}     |      {v_acc:7.2%}      |")
        print("="*50)
        print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))
        
        custom_plot_tune_results('Perceptron', 'T', Ts, pct_tune_results[0], pct_tune_results[1], 'tune_perceptron_T.png')
        custom_plot_tune_results('Avg Perceptron', 'T', Ts, avg_pct_tune_results[0], avg_pct_tune_results[1], 'tune_avg_perceptron_T.png')
        custom_plot_tune_results('Pegasos', 'T', Ts, peg_tune_results_T[0], peg_tune_results_T[1], 'tune_pegasos_T.png')
        custom_plot_tune_results('Pegasos', 'L', Ls, peg_tune_results_L[0], peg_tune_results_L[1], 'tune_pegasos_L.png')

        # Computa acurácia de teste do melhor modelo
        best_pct_T = Ts[np.argmax(pct_tune_results[1])]
        best_avg_pct_T = Ts[np.argmax(avg_pct_tune_results[1])]
        best_peg_T = Ts[np.argmax(peg_tune_results_T[1])]
        best_peg_L = Ls[np.argmax(peg_tune_results_L[1])]

        print("\n--- Computando acurácia no conjunto de teste ---")
        # Test accuracy para o Perceptron
        pct_test_acc = p1.accuracy(p1.classify(test_bow_features, *p1.perceptron(train_bow_features, train_labels, T=best_pct_T)), test_labels)
        print(f"Test accuracy for Perceptron (T={best_pct_T}): {pct_test_acc:.4f}")
        
        # Test accuracy para o Perceptron Médio
        avg_pct_test_acc = p1.accuracy(p1.classify(test_bow_features, *p1.average_perceptron(train_bow_features, train_labels, T=best_avg_pct_T)), test_labels)
        print(f"Test accuracy for Average Perceptron (T={best_avg_pct_T}): {avg_pct_test_acc:.4f}")
        
        # Test accuracy para o Pegasos
        peg_test_acc = p1.accuracy(p1.classify(test_bow_features, *p1.pegasos(train_bow_features, train_labels, T=best_peg_T, L=best_peg_L)), test_labels)
        print(f"Test accuracy for Pegasos (T={best_peg_T}, L={best_peg_L}): {peg_test_acc:.4f}")

        # Obtendo as 10 palavras mais explicativas para classificação positiva
        wordlist = [word for (word, index) in sorted(dictionary.items(), key=lambda x: x[1])]
        best_theta, best_theta_0 = p1.pegasos(train_bow_features, train_labels, T=best_peg_T, L=best_peg_L)
        explanatory_words = utils.most_explanatory_word(best_theta, wordlist)
        print("\n--- 10 unigramas mais explicativos para classificação positiva ---")
        for idx, word in enumerate(explanatory_words[:10]):
            print(f"{idx+1}. Top {idx+1}: {word}")

        #-------------------------------------------------------------------------------
        # Problem 9: Engenharia de Características
        #-------------------------------------------------------------------------------
        print("\n--- Executando Problem 9 (Engenharia de Características) ---")
        
        # 1. Remoção de Stop Words
        dictionary_stop = p1.bag_of_words(train_texts, remove_stopword=True)
        train_stop_features = p1.extract_bow_feature_vectors(train_texts, dictionary_stop, binarize=True)
        test_stop_features = p1.extract_bow_feature_vectors(test_texts, dictionary_stop, binarize=True)
        
        theta_stop, theta_0_stop = p1.pegasos(train_stop_features, train_labels, T=25, L=0.01)
        preds_stop = p1.classify(test_stop_features, theta_stop, theta_0_stop)
        acc_stop = p1.accuracy(preds_stop, test_labels)
        print(f"Acurácia no conjunto de teste (Stop Words Removidas - Binário): {acc_stop:.4f}")
        
        # 2. Características de Contagem (Não-binário)
        train_count_features = p1.extract_bow_feature_vectors(train_texts, dictionary_stop, binarize=False)
        test_count_features = p1.extract_bow_feature_vectors(test_texts, dictionary_stop, binarize=False)
        
        theta_count, theta_0_count = p1.pegasos(train_count_features, train_labels, T=25, L=0.01)
        preds_count = p1.classify(test_count_features, theta_count, theta_0_count)
        acc_count = p1.accuracy(preds_count, test_labels)
        print(f"Acurácia no conjunto de teste (Stop Words Removidas - Contagens): {acc_count:.4f}")

        # 3. Visualização do Espaço Vetorial BoW usando PCA
        print("\n--- Gerando projeção 2D do espaço vetorial BoW (PCA) ---")
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            # Executa a redução de dimensionalidade nas primeiras 500 reviews de treino
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(train_bow_features[:500])
            labels_subset = np.array(train_labels[:500])
            
            plt.figure(figsize=(9, 7))
            colors = ['dodgerblue' if l == 1 else 'darkorange' for l in labels_subset]
            markers = ['^' if l == 1 else 's' for l in labels_subset]
            
            for i in range(len(labels_subset)):
                plt.scatter(features_2d[i, 0], features_2d[i, 1], 
                            color=colors[i], marker=markers[i], alpha=0.75, edgecolors='black', s=60)
            
            plt.title("Projeção 2D (PCA) do Espaço Vetorial Bag of Words\n(Amostra de 500 reviews de treino)", fontsize=12, fontweight='bold', pad=15)
            plt.xlabel("Componente Principal 1", fontsize=10)
            plt.ylabel("Componente Principal 2", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Legenda personalizada
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='dodgerblue', markeredgecolor='black', markersize=10, label='Positiva (+1)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markeredgecolor='black', markersize=10, label='Negativa (-1)')
            ]
            plt.legend(handles=legend_elements, loc='best')
            plt.tight_layout()
            
            filepath = os.path.join('plots', 'bow_space_projection.png')
            plt.savefig(filepath, dpi=150)
            plt.close()
            print(f"Projeção do espaço vetorial salva em '{filepath}'")
            
            # Copia para a pasta de artefatos da conversa atual
            if os.path.exists(ARTIFACT_DIR):
                try:
                    import shutil
                    shutil.copy(filepath, os.path.join(ARTIFACT_DIR, 'bow_space_projection.png'))
                except Exception:
                    pass
        except Exception as e:
            print(f"Erro ao gerar a projeção PCA: {e}")


