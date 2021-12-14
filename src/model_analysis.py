"""
Confronta diverse caratteristiche e modelli.
Divide ogni set di dati in set di addestramento e set di test.
Usa il set di test per stimare le prestazioni in tutti gli esperimenti, dopo
addestra il modello con il set di addestramento. Valuta le prestazioni utilizzando la precisione.
"""

import pandas as pd
import numpy as np
from knn import KNN
from classifier import Classifier
import matplotlib.pyplot as plt
import visualization
from sklearn.model_selection import train_test_split
from preprocessing import Dataset, Preprocessing
from decisiontree import DecisionTree

# -----------------------------------------------------------------------------
# 0. Preprocessing

# ottenere dataframe rielaborati e puliti
cancer_df, hepatitis_df = Preprocessing.get_preprocessed_datasets()
cancer_features, cancer_labels = Preprocessing.get_labels_features(cancer_df)
hepatitis_features, hepatitis_labels = Preprocessing.get_labels_features(hepatitis_df)

# Dataset 1 (Breast Cancer)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    cancer_features, cancer_labels, test_size=0.33
)
# Dataset 2 (Hepatitis)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    hepatitis_features, hepatitis_labels, test_size=0.33
)

# -----------------------------------------------------------------------------
# 1. Confronta la precisione dell'algoritmo KNN e Decision Tree sui due set di dati.


def part1():
    """Valutiamo l'accuratezza del KNN e dell'albero decisionale in base al miglior K e
    iperparametri di profondità trovati (K = 5, profondità = 3 per il cancro e K = 5, profondità = 2 per l'epatite)
    ed esegue il test per 20 volte per risultati più accurati"""

    # KNN (cancer)
    accuracy = Classifier.get_mean_accuracy(
        KNN, X_train_c, X_test_c, y_train_c, y_test_c, 20, k=5
    )
    print("\nKNN accuracy on Breast Cancer Dataset:", accuracy)

    # Decision Tree (cancer)
    accuracy = Classifier.get_mean_accuracy(
        DecisionTree, X_train_c, X_test_c, y_train_c, y_test_c, 20, max_depth=3
    )
    print("Decision Tree accuracy on Breast Cancer Dataset:", accuracy)

    # KNN (hepatitis)
    accuracy = Classifier.get_mean_accuracy(
        KNN, X_train_h, X_test_h, y_train_h, y_test_h, 20, k=5
    )
    print("\nKNN accuracy on Hepatitis Dataset:", accuracy)

    # Decision Tree (hepatitis)
    accuracy = Classifier.get_mean_accuracy(
        DecisionTree, X_train_h, X_test_h, y_train_h, y_test_h, 20, max_depth=2
    )
    print("Decision Tree accuracy on Hepatitis Cancer Dataset:", accuracy)
part1()

# -----------------------------------------------------------------------------
# 2. Prova diversi valori K e guarda come influisce sui dati di allenamento,
#    accuratezza e accuratezza dei dati di test

# Ks che fornisce la massima precisione (li useremo nei passaggi successivi)
hepatitis_k = 5
cancer_k = 5


def part2():
    # Precisione del training
    hepatitis_k = KNN.tune_knn_k(hepatitis_df, training=True)
    cancer_k = KNN.tune_knn_k(cancer_df, training=True)
    print("\nThe ideal K for hepatitis is (based on train accuracy):", hepatitis_k)
    print("The ideal K for breast cancer is (based on train accuracy):", cancer_k)

    # Precisione del testing
    hepatitis_k = KNN.tune_knn_k(hepatitis_df)
    cancer_k = KNN.tune_knn_k(cancer_df)
    print("\nThe ideal K for hepatitis is (based on test accuracy):", hepatitis_k)
    print("The ideal K for breast cancer is (based on test accuracy):", cancer_k)

part2()
# -----------------------------------------------------------------------------
# 3. Verifica in che modo la profondità massima dell'albero può
#  influire sulle prestazioni di Decision Tree sui set di dati forniti.

# Profondità che danno la massima precisione (li useremo nei prossimi passaggi)
hepatitis_d = 2
cancer_d = 3


def part3():
    hepatitis_d = DecisionTree.tune_tree_depth(
        X_train_h, X_test_h, y_train_h, y_test_h, training=True
    )
    cancer_d = DecisionTree.tune_tree_depth(
        X_train_c, X_test_c, y_train_c, y_test_c, training=True
    )
    print("\nThe ideal depth for hepatitis is (based on train accuracy):", hepatitis_d)
    print("The ideal depth for breast cancer is (based on train accuracy):", cancer_d)

    hepatitis_d = DecisionTree.tune_tree_depth(X_train_h, X_test_h, y_train_h, y_test_h)
    cancer_d = DecisionTree.tune_tree_depth(X_train_c, X_test_c, y_train_c, y_test_c)
    print("\nThe ideal depth for hepatitis is (based on test accuracy):", hepatitis_d)
    print("The ideal depth for breast cancer is (based on test accuracy):", cancer_d)

part3()
# -----------------------------------------------------------------------------
# 4. Prova diverse funzioni distanza/costo per entrambi i modelli.
# Trova p ideale nella distanza di minkowski in KNN


def part4():
    hepatitis_p = KNN.tune_knn_p(hepatitis_df)
    cancer_p = KNN.tune_knn_p(cancer_df)
    print("\nThe ideal P for hepatitis minkowski distance function:", hepatitis_p)
    print("The ideal P for breast cancer minkowski distance function:", cancer_p)

    hepatitis_cf = DecisionTree.tune_costfn(
        X_train_h, X_test_h, y_train_h, y_test_h, hepatitis_d
    )
    print(
        "\nThe most accurate cost function for hepatitis dataset:",
        hepatitis_cf.__name__,
    )
    cancer_cf = DecisionTree.tune_costfn(
        X_train_c, X_test_c, y_train_c, y_test_c, cancer_d
    )
    print(
        "\nThe most accurate cost function for breast cancer  dataset:",
        cancer_cf.__name__,
    )
part4()

# -----------------------------------------------------------------------------
# 5. Presenta un grafico del confine decisionale per ciascun modello.


def reduce_df(df, n):
    """Riduce le caratteristiche di un dataframe"""
    reduced_df = df.copy(deep=True)
    corr = visualization.correlation(df, plot=False)
    corr = corr.keys()[:n]
    reduced_df = reduced_df[corr]
    return reduced_df


def part5():
    """Prendiamo 2 caratteristiche con correlazione di alta classe per mostrare i confini delle decisioni
    per il cancro al seno: Uniformity_of_Cell_Shape e Uniformity_of_Cell_Size"""
    # Dataset 1
    # KNN
    df = reduce_df(cancer_df, 3)
    cancer_features, cancer_labels = Preprocessing.get_labels_features(df)

    KNN.plot_decision_bound(
        cancer_features,
        cancer_labels,
        df.keys()[1],
        df.keys()[2],
        KNN,
        k=cancer_k,
    )
    # Decision Tree
    DecisionTree.plot_decision_bound(
        cancer_features,
        cancer_labels,
        df.keys()[1],
        df.keys()[2],
        DecisionTree,
        max_depth=cancer_d,
    )

    # Dataset 2
    # KNN
    df = reduce_df(hepatitis_df, 3)
    hepatitis_features, hepatitis_labels = Preprocessing.get_labels_features(df)

    KNN.plot_decision_bound(
        hepatitis_features,
        hepatitis_labels,
        df.keys()[1],
        df.keys()[2],
        KNN,
        k=hepatitis_k,
    )
    # Decision Tree
    DecisionTree.plot_decision_bound(
        hepatitis_features,
        hepatitis_labels,
        df.keys()[1],
        df.keys()[2],
        DecisionTree,
        max_depth=hepatitis_d,
    )

part5()
# -----------------------------------------------------------------------------
# esperimento extra (part6)
# Esaminiamo l'effetto della riduzione del numero di funzionalità con cui formiamo i modelli,
# quindi tracciamo l'accuratezza del modello addestrato con "n" funzionalità correlate
# alla maggior parte delle classi per vedere se vale la pena considerare tutte le funzionalità.


def reduced_df_accuracy(df, model, **kwargs):
    accuracy = []

    ns = np.arange(2, df.shape[1])
    for n in ns:
        df = reduce_df(df, n)
        features, labels = Preprocessing.get_labels_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33
        )
        accuracy.append(
            model.get_mean_accuracy(
                model, X_train, X_test, y_train, y_test, 1, **kwargs
            )
        )

    plt.plot(ns, accuracy, linewidth=4, markersize=10)
    plt.grid()
    plt.xlabel("Number of features given to the model")
    plt.ylabel("Accuracy")
    plt.show()


def part6():
    reduced_df_accuracy(cancer_df, KNN, k=cancer_k)
    reduced_df_accuracy(cancer_df, DecisionTree, max_depth=cancer_d)

    reduced_df_accuracy(hepatitis_df, KNN, k=hepatitis_k)
    reduced_df_accuracy(hepatitis_df, DecisionTree, max_depth=hepatitis_d)
part6()

def part7():
    """Analizziamo la relazione tra la distribuzione in classi e le caratteristiche
     come TP, TF, FP e FN e la sensibilità. Idealmente vogliamo un modello con alta
     sensibilità"""
    # confusion matrix
    # KNN
    Classifier.plot_confusion_matrix(cancer_features, cancer_labels, KNN, 2, 4, k=5)
    Classifier.plot_confusion_matrix(
        hepatitis_features, hepatitis_labels, KNN, 1, 2, k=5
    )

    # Decision Tree
    Classifier.plot_confusion_matrix(
        cancer_features, cancer_labels, DecisionTree, 2, 4, max_depth=3
    )
    Classifier.plot_confusion_matrix(
        hepatitis_features, hepatitis_labels, DecisionTree, 1, 2, max_depth=2
    )
part7()