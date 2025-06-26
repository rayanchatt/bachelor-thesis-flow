import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, re


path8 = "/Users/macbook/Downloads/Bachelorarbeit/output_combined_8/matched_boxes_8.csv" 
path6 = "output_combined_6/matched_boxes_6.csv" 

def load_iou(csv_path, label):
    df = pd.read_csv(csv_path)
    iou_col = next(c for c in df.columns if c.lower().startswith("iou"))
    return pd.DataFrame({
        "IoU":   df[iou_col].astype(float).dropna(),
        "Dataset": label
    })

df8 = load_iou(path8, "8er (Z1–Z2)")
df6 = load_iou(path6, "6er (Z2–Z3)")
df  = pd.concat([df8, df6], ignore_index=True)

plt.figure(figsize=(5.5,4))
sns.boxplot(x="Dataset", y="IoU", data=df,
            palette=["#5ab4ac", "#d8b365"], width=0.5,
            showfliers=False)
sns.stripplot(x="Dataset", y="IoU", data=df,
              color="k", alpha=.35, size=3, jitter=0.25)
plt.ylim(0,1)
plt.ylabel("IoU"); plt.xlabel("")
plt.title("IoU-Verteilung der gematchten Bounding-Boxen")
plt.tight_layout()
plt.savefig("iou_boxplot.png", dpi=300)
plt.show()