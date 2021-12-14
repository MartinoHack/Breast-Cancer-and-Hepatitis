from collections import Counter
import matplotlib.pyplot as plt
from classifier import Classifier
import numpy as np
from scipy.spatial import distance


class KNN(Classifier):
    """Classificazione K-NN"""

    def __init__(self, k=1, dist_fn=distance.minkowski, p=2):
        self.k = k
        self.dist_fn = dist_fn
        # p è l'ordine della norma nella distanza di minkowski (2 = euclidea)
        self.p = p

    def fit(self, x, y):
        """memorizza il train data"""
        self.train_x = x
        self.train_y = y

    def predict(self, X_test):
        """Predizione Output"""
        predictions = [self._predict_instance(x_test) for x_test in X_test]
        return np.array(predictions)

    def _predict_instance(self, x):
        distances = [self.dist_fn(x, train_x, self.p) for train_x in self.train_x]
        sorted_dist = np.argsort(distances)

        # ottenere indici del k nearest neighbors
        knn_indices = sorted_dist[: self.k]
        # ottenere etichetta dagli indici del k nearest neighbors
        knn_labels = [self.train_y[index] for index in knn_indices]
        return Counter(knn_labels).most_common(1)[0][0]

    @staticmethod
    def cross_validate(k, n, df, p_dist=2, training=False):
        """Quando il parametro di training è True, l'accuratezza del modello
        è testato sulla base degli esempi su cui è stato costruito"""
        acc = []
        # restituire un campione casuale di articoli
        df.sample(frac=1)
        num_rows = df.shape[0]
        num_folds_row = num_rows // k

        for f in range(k):
            test = df[f * num_folds_row : f * num_folds_row + num_folds_row]
            if training:
                train = df[df.index.isin(test.index)]
            else:
                train = df[~df.index.isin(test.index)]
            train_x, train_y = (
                train.drop("Class", axis=1).to_numpy(),
                train["Class"].to_numpy(),
            )
            test_x, test_y = (
                test.drop("Class", axis=1).to_numpy(),
                test["Class"].to_numpy(),
            )

            classifier_knn = KNN(n, p=p_dist)
            classifier_knn.fit(train_x, train_y)
            predictions = classifier_knn.predict(test_x)
            acc.append(classifier_knn.evaluate_acc(predictions, test_y))
        return np.array(acc).mean()

    @staticmethod
    def tune_knn_k(df, training=False):
        """Utilizzare la convalida incrociata k-fold per ottimizzare l'iperparametro K
        e tracciare la precisione per diversi iperparametri K"""
        scores = []
        ks = np.arange(1, 16)
        folds = 10
        for k in ks:
            accuracy = KNN.cross_validate(folds, k, df, training=training)
            scores.append(accuracy)

        plt.plot(ks, scores, linewidth=4, markersize=10)
        plt.grid()
        plt.xlabel("K in K-nearest Neighbors")
        if training:
            plt.ylabel("Cross Validation Train Accuracy")
        else:
            plt.ylabel("Cross Validation Test Accuracy")
        plt.show()

        # K con maggiore accuratezza
        return np.array(scores).argmax() + 1

    @staticmethod
    def tune_knn_p(df, training=False):
        """Utilizzare la convalida incrociata k-fold per ottimizzare l'iperparametro P
        e tracciare la precisione per diversi iperparametri P"""
        scores = []
        ps = np.arange(1, 16)
        folds = 10
        for p in ps:
            accuracy = KNN.cross_validate(folds, 5, df, p_dist=p)
            scores.append(accuracy)

        plt.plot(ps, scores, linewidth=4, markersize=10)
        plt.grid()
        plt.xlabel("P in K-nearest Neighbors's minkowski distance function")
        if training:
            plt.ylabel("Cross Validation Train Accuracy")
        else:
            plt.ylabel("Cross Validation Test Accuracy")
        plt.show()

        # the p with highest accuracy
        return np.array(scores).argmax() + 1
