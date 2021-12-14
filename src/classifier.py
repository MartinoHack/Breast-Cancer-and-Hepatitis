import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class Classifier:
    @staticmethod
    def evaluate_acc(true_labels, target_labels):
        """Valutare la precisione del modello"""
        accuracy = np.sum(true_labels == target_labels) / len(true_labels)
        return accuracy

    @staticmethod
    def plot_decision_bound(features, labels, f1_name, f2_name, model, **kwargs):

        labels = labels.flatten()
        mod = model(**kwargs)

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33, random_state=1234
        )

        mod.fit(X_train, y_train)
        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
        if type(mod).__name__ == "DecisionTree":
            predictions = np.argmax(
                mod.predict(np.c_[xx.ravel(), yy.ravel()]), 1
            ).reshape(xx.shape)
        else:
            predictions = mod.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, predictions, alpha=0.4)
        plt.scatter(features[:, 0], features[:, 1], c=labels)
        plt.xlabel(f1_name)
        plt.ylabel(f2_name)
        plt.xticks()
        plt.yticks()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(features, labels, model, l1, l2, **kwargs):
        """Traccia la matrice di confusione di un modello. l1 e l2 sono le due classi"""
        mod = model(**kwargs)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33
        )

        mod.fit(X_train, y_train)
        predictions = mod.predict(X_test)
        if type(mod).__name__ == "DecisionTree":
            predictions = np.argmax(predictions, 1)
        TP, TN, FP, FN = 0, 0, 0, 0

        for i in range(len(predictions)):
            if predictions[i] == y_test[i] == l1:
                TP += 1
            elif predictions[i] == y_test[i] == l2:
                TN += 1
            elif predictions[i] != y_test[i] and predictions[i] == l1:
                FP += 1
            else:
                FN += 1
        confusion_matrix = [[TP, FP], [FN, TN]]

        sns.heatmap(confusion_matrix, annot=True)
        plt.ylabel("Predicted Class")
        plt.xlabel("True Class")
        plt.yticks(ticks=[0.5, 1.5], labels=[l1, l2])
        plt.xticks(ticks=[0.5, 1.5], labels=[l1, l2])
        plt.title("Confusion matrix")
        plt.show()

    @staticmethod
    def get_mean_accuracy(
        model, X_train, X_test, y_train, y_test, n, training=False, **kwargs
    ):
        """restituisce l'accuratezza di KNN prendendo l'accuratezza media di n iterazioni"""
        accuracy = []
        for _ in range(n):
            mod = model(**kwargs)
            mod.fit(X_train, y_train)
            if training:
                X_test = X_train
                y_test = y_train
            predictions = mod.predict(X_test)
            if type(mod).__name__ == "DecisionTree":
                predictions = np.argmax(predictions, 1)
            accuracy.append(mod.evaluate_acc(predictions, y_test))
        return np.array(accuracy).mean()
