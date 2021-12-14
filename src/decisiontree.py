import numpy as np
import matplotlib.pyplot as plt
from classifier import Classifier

np.random.seed(1234)


class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices  # memorizza gli indici di dati che si trovano nella regione definita da questo nodo
        self.left = None  # memorizza il figlio sinistro del nodo
        self.right = None  # memorizza il figlio destro del nodo
        self.split_feature = None  # la funzione per dividere in questo nodo
        self.split_value = None  # il valore della caratteristica per la divisione in questo nodo
        if parent:
            self.depth = (
                parent.depth + 1
            )  # ottenere la profondità  del nodo aggiungendo uno alla profondità del genitore
            self.num_classes = (
                parent.num_classes
            )  # copia le classi num dall'elemento padre
            self.data = parent.data  # copia i dati dall'elemento padre
            self.labels = parent.labels  # copia le etichette dal padre
            class_prob = np.bincount(
                self.labels[data_indices], minlength=self.num_classes
            )  # sta contando la frequenza di diverse etichette nella regione definita da questo nodo
            self.class_prob = class_prob / np.sum(
                class_prob
            )  # memorizza la probability class per il nodo


def greedy_test(node, cost_fn):
    #
    # inizializzare i migliori valori dei parametri
    best_cost = np.inf
    best_feature, best_value = None, None
    num_instances, num_features = node.data.shape
    # ordinare le caratteristiche per ottenere i candidati del valore di test prendendo la media dei valori delle caratteristiche ordinate consecutive
    data_sorted = np.sort(node.data[node.data_indices], axis=0)
    test_candidates = (data_sorted[1:] + data_sorted[:-1]) / 2.0
    for f in range(num_features):
        # memorizza i dati corrispondenti alla funzione f-esima
        data_f = node.data[node.data_indices, f]
        for test in test_candidates[:, f]:
            # Divide gli indici usando il valore di prova della funzione f-esima
            left_indices = node.data_indices[data_f <= test]
            right_indices = node.data_indices[data_f > test]
            # non possiamo avere una divisione in cui un figlio ha zero elementi
            #
            # se questo è vero per tutte le caratteristiche di test e i loro valori di test, la funzione restituisce il miglior costo come infinito
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            # calcola il costo sinistro e destro in base alla divisione corrente
            left_cost = cost_fn(node.labels[left_indices])
            right_cost = cost_fn(node.labels[right_indices])
            num_left, num_right = left_indices.shape[0], right_indices.shape[0]
            # ottenere il costo combinato utilizzando la somma ponderata del costo sinistro e destro
            cost = (num_left * left_cost + num_right * right_cost) / num_instances
            # aggiornare solo quando si riscontra un costo inferiore
            if cost < best_cost:
                best_cost = cost
                best_feature = f
                best_value = test
    return best_cost, best_feature, best_value


# calcola il costo di errata classificazione sottraendo la probabilità massima di qualsiasi classe
def cost_misclassification(labels):
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)


#
# calcola l'entropia delle etichette calcolando la probability class
def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[
        class_probs > 0
    ]
    return -np.sum(
        class_probs * np.log(class_probs)
    )  # espressione per l'entropia -\sigma p(x)log[p(x)]


# calcola il costo dell'indice Gini
def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(
        np.square(class_probs)
    )  # espressione per indice Gini 1-\sigma p(x)^2


class DecisionTree(Classifier):
    def __init__(
        self,
        num_classes=None,
        max_depth=3,
        cost_fn=cost_misclassification,
        min_leaf_instances=1,
    ):
        self.max_depth = max_depth  # profondità massima per la terminazione
        self.root = None  # memorizza la radice dell'albero decisionale
        self.cost_fn = cost_fn  # memorizza la funzione di costo dell'albero decisionale
        self.num_classes = num_classes  # memorizza il numero totale di classi
        self.min_leaf_instances = (
            min_leaf_instances  # numero minimo di istanze in una foglia per la terminazione
        )

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        if self.num_classes is None:
            self.num_classes = np.max(labels) + 1
        # inizializzazione della radice dell'albero decisionale
        self.root = Node(np.arange(data.shape[0]), None)
        self.root.data = data
        self.root.labels = labels
        self.root.num_classes = self.num_classes
        self.root.depth = 0
        # per costruire ricorsivamente il resto dell'albero
        self._fit_tree(self.root)

    def predict(self, data_test):
        class_probs = np.zeros((data_test.shape[0], self.num_classes))
        for n, x in enumerate(data_test):
            node = self.root
            # ciclo lungo la profondità della regione ad albero in cui rientra il campione di dati presente in base alla funzione e al valore di suddivisione split
            while node.left:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            # il ciclo termina quando raggiungi una foglia dell'albero e la probability class di quel nodo viene presa per la previsione
            class_probs[n, :] = node.class_prob
        return class_probs

    def _fit_tree(self, node):
        # Questo dà la condizione per la terminazione della ricorsione risultante in un nodo foglia
        if (
            node.depth == self.max_depth
            or len(node.data_indices) <= self.min_leaf_instances
        ):
            return
        # selezionare il miglior test minimizzando il costo
        cost, split_feature, split_value = greedy_test(node, self.cost_fn)
        # se il costo restituito è infinito significa che non è possibile dividere il nodo e quindi terminare
        if np.isinf(cost):
            return
        # per ottenere un array booleano che suggerisca quali indici di dati corrispondenti a questo nodo si trovano a sinistra della divisione
        test = node.data[node.data_indices, split_feature] <= split_value
        # memorizzare la funzione di divisione e il valore del nodo
        node.split_feature = split_feature
        node.split_value = split_value
        # definire nuovi nodi che saranno il figlio sinistro e destro del nodo attuale
        left = Node(node.data_indices[test], node)
        right = Node(node.data_indices[np.logical_not(test)], node)
        # chiamata ricorsiva a _fit_tree()
        self._fit_tree(left)
        self._fit_tree(right)
        # assegna il figlio sinistro e destro al figlio presente
        node.left = left
        node.right = right

    @staticmethod
    def tune_tree_depth(X_train, X_test, y_train, y_test, training=False):
        scores = []
        ds = np.arange(1, 20)
        for d in ds:
            accuracy = Classifier.get_mean_accuracy(
                DecisionTree,
                X_train,
                X_test,
                y_train,
                y_test,
                5,
                training=training,
                max_depth=d,
            )
            scores.append(accuracy)

        plt.plot(ds, scores, linewidth=4, markersize=10)
        plt.grid()
        plt.xlabel("Depth in decision tree")
        if training:
            plt.ylabel("Cross Validation Train Accuracy")
        else:
            plt.ylabel("Cross Validation Test Accuracy")
        plt.show()

        # la profondità con la massima precisione
        return np.array(scores).argmax() + 1

    @staticmethod
    def tune_costfn(X_train, X_test, y_train, y_test, depth):
        scores = []
        cost_fns = [cost_misclassification, cost_entropy, cost_gini_index]
        for cost_fn in cost_fns:
            accuracy = Classifier.get_mean_accuracy(
                DecisionTree,
                X_train,
                X_test,
                y_train,
                y_test,
                5,
                max_depth=depth,
                cost_fn=cost_fn,
            )
            scores.append(accuracy)

        print(scores)
        plt.bar(x=[1, 2, 3], height=scores)
        plt.xlabel("Cost function")
        plt.xticks(
            ticks=[1, 2, 3],
            labels=["cost misclassification", "cost entropy", "cost gini index"],
        )
        plt.ylabel("Accuracy")
        plt.show()

        # la profondità con la massima precisione
        return cost_fns[np.array(scores).argmax()]
