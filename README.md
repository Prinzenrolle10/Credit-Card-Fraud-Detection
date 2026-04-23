# Advanced Credit Card Fraud Detection: XGBoost vs. Deep Learning

**Projektfokus:** Data Science, Imbalanced Data Handling, Hyperparameter-Tuning, Modell-Evaluation.

Dieses Projekt widmet sich der Erkennung von Kreditkartenbetrug anhand stark unbalancierter Transaktionsdaten. Das Ziel war es, den Business Value zu maximieren, indem möglichst viele Betrugsfälle erkannt werden, ohne dabei das Kundenerlebnis durch falsche Alarme (False Positives) zu stören. 

Dabei wurde empirisch untersucht, ob moderne Deep Learning Ansätze (Autoencoder) klassische baumbasierte Ensembles (XGBoost) auf PCA-komprimierten Tabellendaten schlagen können.

## Tech Stack
* **Sprache:** Python 3.10
* **Machine Learning:** XGBoost, Scikit-Learn
* **Deep Learning:** TensorFlow / Keras
* **AutoML & Tuning:** Optuna
* **Umgebung:** Google Colab (GPU-beschleunigtes Training)

## Datengrundlage
Der verwendete Datensatz ist der offizielle **[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** von Kaggle, bereitgestellt von der Machine Learning Group (MLG) der Université Libre de Bruxelles (ULB) sowie dem Worldline and the Government of the Brussels-Capital Region.

* **Umfang:** 284.807 echte Transaktionen europäischer Kreditkarteninhaber aus dem September 2013.
* **Klassenverteilung:** Hochgradig unbalanciert mit lediglich 492 bestätigten Betrugsfällen (0,172 %).
* **Besonderheit (Datenschutz):** Um die Privatsphäre der Kunden zu schützen, wurden die ursprünglichen Features (V1 bis V28) vorab durch eine Principal Component Analysis (PCA) mathematisch transformiert und dekorreliert. Lediglich die Spalten `Time` und `Amount` (Betrag) blieben im Originalzustand.

## Methodik und Architektur

1. **Robustes Data Engineering:**
   * Generierung zeitbasierter Features (z. B. rollierende Transaktionsfenster).
   * Vermeidung von Data Leakage durch strikt chronologisches Train-Validation-Test-Splitting (60-20-20).
   * Vollständige Skalierung (`StandardScaler`) aller Features als Vorbereitung für das Neuronale Netz.

2. **Der ML-Pfad (XGBoost):**
   * Bayesian Optimization der Hyperparameter via **Optuna** auf der GPU.
   * Dynamische Gewichtung der Minderheitsklasse (`scale_pos_weight`) zur Adressierung der Class Imbalance.
   * Strikte Optimierung auf den **AUCPR** (Area Under the Precision-Recall Curve).

3. **Der DL-Pfad (Autoencoder):**
   * Aufbau eines Neuronalen Netzes zur unüberwachten Anomalieerkennung (Semi-supervised Learning).
   * Das Netz wurde ausschliesslich auf legitimen Transaktionen trainiert. Betrugsfälle sollten durch einen massiven Anstieg des Rekonstruktionsfehlers (Mean Squared Error) identifiziert werden.

## Ergebnisse

Das Experiment lieferte ein eindeutiges Ergebnis zugunsten des klassischen Machine Learnings:
* **Gewinner:** XGBoost (Validation AUCPR: ~0.788)
* **Verlierer:** Autoencoder (Validation AUCPR: ~0.057)

**Warum hat Deep Learning hier versagt?**
Die V-Features des Kaggle-Datensatzes wurden vorab durch eine Principal Component Analysis (PCA) anonymisiert und dekorreliert. Ein Autoencoder basiert jedoch darauf, Korrelationen in den Daten zu finden, um diese im "Bottleneck" komprimieren zu können. Durch die PCA wurde dem Netz diese Informationsgrundlage entzogen. Entscheidungsbäume (XGBoost) hingegen suchen nicht nach Korrelationen, sondern ziehen harte Grenzen im Feature-Raum, was sie hier überlegen machte.

### Finale Performance (Business Case)
Durch ein gezieltes Tuning des Klassifizierungsschwellenwerts (Threshold-Optimierung via F1-Score-Maximierung) konnte das XGBoost-Modell perfekt auf den Geschäftsnutzen ausbalanciert werden.

Auf den **komplett ungesehenen Testdaten (über 56.000 Transaktionen)** erreichte das Modell folgende Werte:
* **55 von 75 echten Betrugsfällen (73 %)** wurden erfolgreich gestoppt.
* **Nur 7 legitime Kunden (False Positives)** wurden fälschlicherweise blockiert (Precision von 89 %).
