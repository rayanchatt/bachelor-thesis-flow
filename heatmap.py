"""
heatmap.py

Erzeugt jeweils eine 2D-KDE-Heatmap der Mittelpunkte aus:
  – matched_boxes_6.csv (6er-Datensatz: Z2/Z3)
  – matched_boxes_8.csv (8er-Datensatz: Z1/Z2)

Vorbereitung:
    pip install pandas seaborn matplotlib
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

# 1) Pfade zu den CSV-Dateien (anpassen):
PATH_6 = "/Users/macbook/Downloads/Bachelorarbeit/output_combined_6/matched_boxes_6.csv"
PATH_8 = "/Users/macbook/Downloads/Bachelorarbeit/output_combined_8/matched_boxes_8.csv"

# 2) Hilfsfunktion: Parst einen String "(x, y, w, h)" und gibt den Mittelpunkt (x_center, y_center) zurück.
def parse_center(box_str):
    """
    box_str: Beispiel "(123.0, 456.0, 50.0, 40.0)"
    Rückgabe: (x_center, y_center)
    """
    x, y, w, h = ast.literal_eval(box_str)
    return x + w/2, y + h/2

# 3) Daten für den 6er-Datensatz einlesen und Mitte berechnen
if not os.path.exists(PATH_6):
    raise FileNotFoundError(f"Datei nicht gefunden: {PATH_6}")

df6 = pd.read_csv(PATH_6)
# Wir nehmen hier 'green_box' (Box aus Z2), kann aber auch 'red_box' sein – beide liegen auf denselben Match-Koordinaten.
centers_6 = df6["green_box"].apply(parse_center).tolist()
df6_centers = pd.DataFrame(centers_6, columns=["x_center", "y_center"])

# 4) Daten für den 8er-Datensatz einlesen und Mitte berechnen
if not os.path.exists(PATH_8):
    raise FileNotFoundError(f"Datei nicht gefunden: {PATH_8}")

df8 = pd.read_csv(PATH_8)
centers_8 = df8["green_box"].apply(parse_center).tolist()
df8_centers = pd.DataFrame(centers_8, columns=["x_center", "y_center"])

# 5) Plot für den 6er-Datensatz
plt.figure(figsize=(7,7))
sns.kdeplot(
    data=df6_centers,
    x="x_center",
    y="y_center",
    levels=10,
    cmap="Oranges",
    fill=True,
    alpha=0.6,
    thresh=0.05
)
plt.title("Heatmap der Mittelpunkte (6er-Datensatz: Z₂ / Z₃)", fontsize=14)
plt.xlabel("x-Mittelpunkt (Pixel)")
plt.ylabel("y-Mittelpunkt (Pixel)")
plt.tight_layout()
plt.savefig("heatmap_6er.png", dpi=300)
plt.show()

# 6) Plot für den 8er-Datensatz
plt.figure(figsize=(7,7))
sns.kdeplot(
    data=df8_centers,
    x="x_center",
    y="y_center",
    levels=10,
    cmap="Blues",
    fill=True,
    alpha=0.6,
    thresh=0.05
)
plt.title("Heatmap der Mittelpunkte (8er-Datensatz: Z₁ / Z₂)", fontsize=14)
plt.xlabel("x-Mittelpunkt (Pixel)")
plt.ylabel("y-Mittelpunkt (Pixel)")
plt.tight_layout()
plt.savefig("heatmap_8er.png", dpi=300)
plt.show()