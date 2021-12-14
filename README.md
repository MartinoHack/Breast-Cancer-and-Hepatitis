# Breast-Cancer-and-Hepatitis
Prediction of Breast Cancer and Hepatitis in Python.


Negli ultimi anni, sono stati sviluppati modelli di apprendimento automatico per identificare le malattie
principalmente in base ai loro sintomi.
In questo progetto, implementiamo due diverse tecniche di classificazione (K-Nearest Neighbor e Decision
Tree) per il cancro al seno e l'epatite. I due classificatori sono stati formati e verificati utilizzando due
dataset sanitari di riferimento, set di dati sul cancro (Breast Cancer Wisconsin) e sull'epatite (Hepatitis). La
classificazione per entrambe le malattie include livelli benigni o maligni.
I due dataset sono stati preelaborati prima di essere pronti per l’addestramento, verifica e test.
-La fase di pre-elaborazione includeva la normalizzazione dei dati e la rimozione di caratteristicheirrilevanti (come l'IDdel paziente), istanze duplicate e istanze con caratteristiche mancanti;
-E’ stata condotta una fase di analisi sul set di dati per migliorare la comprensione.
Dopo la preelaborazione e l'analisi di entrambi i set di dati, i modelli K-Nearest Neighbor e Decision Treesono stati implementati da zero. I modelli sono stati quindi sottoposti a convalida incrociata per ottimizzareiperparametri come K inK-Nearest Neighbor e profondità massima in Decision Tree e per trovare la funzionedi distanza ideale per K-Nearest Neighbor e la funzione di costo per Decision Tree per ciascun set di dati.
I risultati della nostra analisi comparativa mostrano che il classificatore Decision Tree superaleggermente il classificatore K-Nearest Neighbor sul set di dati sul cancro al seno e conun'elevata differenza sul set di datisull'epatite.
I due set di dati su cui conduciamo la nostra ricerca sono distinti e simili in molti modi. Ad esempio,entrambi contengono una distribuzione delle classi benigna rispetto a quella maligna molto disomogenea.Tuttavia, la differenza più significativa tra il cancro al seno e il set di dati sull'epatite è il numero di puntidati e le caratteristiche che contengono. Questo ci consente di identificare e confrontare i punti di forza edi debolezza dei nostri modelli. Dopo la preelaborazione, il set di dati sull'epatite, che contiene un grannumero di caratteristiche rispetto al set di dati sul cancro al seno, viene lasciato con un piccolo numero dipunti dati.

Dataset
I due set di dati, vale a dire il cancro al seno e l'epatite hanno dimensione 700 × 11 e 156 × 20 rispettivamente. Il cancro al seno contiene caratteristiche numeriche mentre l'epatite include caratteristiche sia numeriche che categoriali. Secondo la descrizione dei dati del cancro al seno, le etichette di classe sono associate a tumori benigni e maligni. Le etichette del
set di dati sull'epatite sono divise in due classi "die" e "live". Entrambi i set di dati contengono una distribuzione di classi non uniforme.
Per preelaborare entrambi i set di dati, viene definita una classe di preelaborazione, in cui vengono rimosse le informazioni mancanti. Quindi le probabili righe duplicate, vuote e malformate vengono omesse tramite una funzione di pulizia.
Viene rimossa una percentuale considerevole dal set di dati sull'epatite, circa il 48%, mentre solo il 3,46% dal set di dati sul cancro al seno viene eliminato attraverso il processo di pulizia. Infine, entrambi i dataset vengono dichiarati in un formato unificato (l'etichetta della classe viene portata nella prima colonna, le funzionalità seguono nelle colonne successive). Le caratteristiche sono normalizzate nella funzione correlata tramite il ridimensionamento “minmax” implementato manualmente.
Per acquisire una migliore comprensione dei set di dati, viene distinto il numero di casi in ciascuna classe. Viene visualizzata la proporzione di etichette maligne e benigne, oltre a quelle morte e vive.
Un esempio può essere visto nella Figura 1 per il set di dati sul cancro al seno. L'analisi viene eseguita utilizzando istogrammi e statistiche su set di dati.
Gli istogrammi si trovano per illustrare la distribuzione della frequenza di classe e caratteristica all'interno di entrambi i campioni di set di dati.
La Figura 1 può rappresentare chiaramente caratteristiche fortemente correlate con l'obiettivo nel set di dati sul cancro al seno. Le caratteristiche con correlazione di alta classe sono state selezionate nella speranza di ottenere una maggiore precisione.
