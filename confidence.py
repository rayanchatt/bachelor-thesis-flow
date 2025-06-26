import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, glob, os, re

base_dir = "CodeExampleRayan/yolov5/runs/detect"

frames = []
for fn in glob.glob(f"{base_dir}/**/predictions_*_6.csv", recursive=True):
    m = re.search(r"predictions_(\d)_6\.csv", os.path.basename(fn))
    if not m:
        continue
    layer = f"Z{m.group(1)}" 
    df = pd.read_csv(fn, usecols=['Confidence'])
    df['Layer'] = layer
    frames.append(df)

if not frames:
    raise RuntimeError("Keine *_6.csv-Dateien gefunden – Pfad prüfen!")

conf = pd.concat(frames, ignore_index=True)

plt.figure(figsize=(6,4))
sns.kdeplot(data=conf, x='Confidence', hue='Layer',
            fill=True, alpha=.4, bw_adjust=1.1)
plt.xlim(0,1)
plt.title("Confidence-Verteilung: 6er-Datensatz (Z2 & Z3)")
plt.tight_layout()
plt.savefig("confidence_kde_6_only.png", dpi=300)
plt.show()