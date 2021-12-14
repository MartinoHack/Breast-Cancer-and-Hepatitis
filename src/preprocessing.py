import pandas as pd
from enum import Enum


class Dataset(Enum):
    breast_cancer = 1
    hepatitis = 2


class Preprocessing:
    """Acquisizione, Analisi e Pulizia dati

    Abbiamo utilizzato i seguenti dataset:
    - Dataset 1: breast cancer wisconsin.csv (Breast Cancer dataset):
         https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    - Dataset 2: hepatitis.csv (Hepatitis dataset):
         http://archive.ics.uci.edu/ml/datasets/Hepatitis
    """

    DATASET1 = "../data/breast_cancer_wisconsin.csv"
    DATASET2 = "../data/hepatitis.csv"

    def __init__(self, dataset):
        """acquizione dati"""
        # caricare i dataset nel dataframe di pandas
        if dataset == Dataset.breast_cancer:
            self.df = pd.read_csv(Preprocessing.DATASET1, na_values="?")
        elif dataset == Dataset.hepatitis:
            self.df = pd.read_csv(Preprocessing.DATASET2, na_values="?")
        else:
            raise IndexError("Dataset enumeration between 1 and 2")

    def clean(self):
        print("Cleaning dataset ...")
        self.df = self._remove_duplicate_rows()
        self.df = self._remove_empty_rows()

        # elimina le colonne id non rilevanti
        self.df.drop("id", axis=1, inplace=True, errors="ignore")
        print("\nDropped irrelevant features")

        # imposta la colonna della classe in prima posizione
        class_col = self.df.pop("Class")
        self.df.insert(0, "Class", class_col)
        print("Set class column to the first position")

        self.df.reset_index(drop=True, inplace=True)
        print("Reset indices")

    def _remove_empty_rows(self):
        """rimuovere le righe contenenti una colonna con un valore vuoto"""
        clean_df = self.df.dropna()
        rows_removed = Preprocessing._rows_removed(self.df, clean_df)
        percentage_removed = Preprocessing._percentage_removed(rows_removed, self.df)
        print(
            "\nRemoved empty rows: \nRemoved {percentage}% of rows \n{count} rows removed".format(
                percentage=percentage_removed, count=rows_removed
            )
        )
        return clean_df

    def _remove_duplicate_rows(self):
        """rimuove le righe con tutte le funzioni duplicate"""
        clean_df = self.df.drop_duplicates()
        rows_removed = Preprocessing._rows_removed(self.df, clean_df)
        percentage_removed = Preprocessing._percentage_removed(rows_removed, self.df)
        print(
            "\nRemoved duplicate rows: \nRemoved {percentage}% of rows \n{count} rows removed".format(
                percentage=percentage_removed, count=rows_removed
            )
        )
        return clean_df

    def normalize_features(self):
        class_col = self.df.pop("Class")
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        self.df.insert(0, "Class", class_col)

    _rows_removed = lambda df1, df2: abs(df1.shape[0] - df2.shape[0])
    _percentage_removed = lambda removed, df: round(100 * removed / df.shape[0], 2)

    @staticmethod
    def get_preprocessed_datasets(normalize_features=True):
        # Carica set di dati in un preprocessore
        cancer_preprocessor = Preprocessing(Dataset.breast_cancer)
        hepatitis_preprocessor = Preprocessing(Dataset.hepatitis)

        # Pulizia dei dati
        print("Cleaning hepatitis dataset")
        hepatitis_preprocessor.clean()

        print("\nCleaning breat cancer dataset")
        cancer_preprocessor.clean()
        if normalize_features:
            hepatitis_preprocessor.normalize_features()
            cancer_preprocessor.normalize_features()
        return cancer_preprocessor.df, hepatitis_preprocessor.df

    @staticmethod
    def get_labels_features(df):
        """separa le etichette e le caratteristiche in un set di dati e le restituisce"""
        dframe = df.copy(deep=True)
        labels = dframe.pop("Class").to_numpy()
        features = dframe.to_numpy()
        return features, labels
