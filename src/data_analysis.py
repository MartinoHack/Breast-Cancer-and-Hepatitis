from preprocessing import Preprocessing
import visualization as vis


def print_stats(df):
    shape = df.shape
    print(
        "\nDataset contains {rows} rows and {columns} features".format(
            rows=shape[0], columns=shape[1] - 1
        )
    )
    print(df.describe())


cancer_df, hepatitis_df = Preprocessing.get_preprocessed_datasets()

# Analisi invariate
# -----------------------------------------------------------------------------

# stampa statistiche di base sui set di dati
print("breat cancer dataset")
print_stats(cancer_df)
print("hepatitis dataset")
print_stats(hepatitis_df)

# distribuzione dei casi positivi vs negativi
vis.class_countplot(cancer_df)
vis.class_countplot(hepatitis_df)

# disegna istogrammi
vis.feature_hist(cancer_df)
vis.feature_hist(hepatitis_df)

# Analisi multivariata
# -----------------------------------------------------------------------------

print("Cancer dataset most decisive features", vis.correlation(cancer_df))
print("Hepatitis dataset most decisive features", vis.correlation(hepatitis_df))
