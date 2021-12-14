from preprocessing import Dataset, Preprocessing
import pandas as pd
from pandas.plotting import parallel_coordinates as pc
import matplotlib.pyplot as plt
import seaborn as sns


def class_countplot(df):
    """Frequenze e conteggio della classe di stampa"""

    perc = df["Class"].value_counts(normalize=True) * 100
    perc1, perc2 = "{:.2f}".format(perc.iloc[0]), "{:.2f}".format(perc.iloc[1])
    msg = "Percentages: {perc1}% vs {perc2}%".format(perc1=perc1, perc2=perc2)
    sns.countplot(x="Class", data=df)
    plt.title("Class frequency table\n" + msg)
    plt.ylabel("Number of cases per class")
    plt.xlabel("Classes")
    plt.show()


def feature_hist(df):
    df.hist(grid=False)
    plt.title("Feature histograms")
    plt.show()


def feature_scatter_box(x, df):
    sns.swarmplot(y=df["Class"], x=df[x], color="0.2", orient="h")
    sns.boxplot(y=df["Class"], x=df[x], orient="h")
    plt.show()


# Analisi multivariata
def correlation(df, plot=True):
    """Heatmap (matrice di correlazione a coppie)"""

    # trova le caratteristiche più correlate alla classe
    correlation = df.corrwith(df["Class"]).abs().sort_values(ascending=False)
    if plot:
        heatmap = df.corr()
        sns.heatmap(heatmap, annot=True, cmap="coolwarm")
        plt.title("Correlation matrix")
        plt.show()

    return correlation


def parallel_coordinates(df):
    """Grafico coordinate parallele"""

    frame = df.apply(lambda x: x / x.max(), axis=0)
    pc(frame, "Class", color=("#FF1493", "#008000"))
    plt.show()


def three_d_plot(x, y, z, df):
    ax = plt.axes(projection="3d")
    ax.scatter(
        xs=df[x], ys=df[y], zs=df[z], c=df["Class"], cmap="viridis", linewidth=0.5
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()
