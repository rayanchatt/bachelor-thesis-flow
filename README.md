# Quantifizierung & Modellierung von Gewebewachstum  
*CPU-basierte Pipeline für Mitose-Detektion & Divergenz-Analyse*

Dieses Repository enthält sämtliche Quellcodes der Bachelorarbeit  
**„Quantifizierung und Modellierung von Gewebewachstum“**  
(Heinrich-Heine-Universität Düsseldorf, 2025).  

Ziel ist eine **GPU-freie** Auswertungskette, die

1. Mitose-Ereignisse in Hellfeld-Zeitreihen mit einem schlanken **YOLOv5s**-Detektor erkennt und  
2. deren mechanisches Umfeld über **dichte Optical-Flow-Felder** (Farnebäck) als Divergenz-/Magnitude-Karten (*DefMaps*) quantifiziert.

> **Kurzübersicht**  
> – ~95 % mAP<sub>50</sub> auf der Trainingsebene  
> – Signifikante Divergenz-Minima vor und Divergenz-Maxima nach der Mitose  

---

## Ordner & Dateien

| Pfad / Datei                     | Zweck |
|---------------------------------|-------|
| `build_defmap_stack.py`          | Erzeugt Divergenz-/Magnitude-Stacks aus Optical-Flow‐Feldern. |
| `confidence.py`                  | Berechnet adaptive Confidence-Schwelle via KDE & ROC. |
| `heatmap.py`                     | Rendert Dichte-Heatmaps der Mitosepositionen. |
| `iou.py`                         | Hilfsfunktionen für Intersection-over-Union-Matching in Z-Richtung. |
| `main_example_images.py`         | Minimalbeispiel zur Pipeline-Demo auf Sample-Frames. |
| `match_labels.py`                | Verknüpft Bounding-Boxen temporär & axial, weist Track-IDs zu. |
| `rgbcode.py`                     | Stapelt drei Z-Schichten zu RGB-Tensor-Frames. |
| `timelagcorrection_div.py`       | Lag-Analyse: Confidence ↔ Divergenz. |
| `timelagcorrection_mag.py`       | Lag-Analyse: Confidence ↔ Betrag der Verschiebung. |

*(Alle Skripte laufen standalone)*

---
