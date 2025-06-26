import os, glob, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse

DEFAULT_DEFMAPS = [
    "/Users/macbook/Downloads/Bachelorarbeit/project/defmap_output_mag_8/defmap_stack_Z1_mag.npy",
    "/Users/macbook/Downloads/Bachelorarbeit/project/defmap_output_mag_8/defmap_stack_Z2_mag.npy",
    "/Users/macbook/Downloads/Bachelorarbeit/project/defmap_output_mag_8/defmap_stack_Z3_mag.npy"
]
DEFAULT_LABEL_DIR = "/Users/macbook/Downloads/Bachelorarbeit/CodeExampleRayan/yolov5/runs/detect/ch3_82/labels"
DEFAULT_OUT_DIR = "/Users/macbook/Downloads/Bachelorarbeit/project/lagcorr_new_mag_8"

parser = argparse.ArgumentParser(
    description="Lag‑Korrelation YOLO‑Confidence ↔ Magnitude")
parser.add_argument("--defmap", nargs="+", default=DEFAULT_DEFMAPS,
                    help="Eine oder mehrere *.npy-Dateien mit Magnitude-Stacks (Default: DEFAULT_DEFMAPS)")
parser.add_argument("--labels", default=DEFAULT_LABEL_DIR,
                    help="Ordner mit YOLO-*.txt Labels (Default: DEFAULT_LABEL_DIR)")
parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                    help="Zielordner für CSV & Plots (Default: DEFAULT_OUT_DIR)")
parser.add_argument("--conf_min", type=float, default=0.5,
                    help="Minimaler Confidence‑Wert (Default 0.5)")

args = parser.parse_args()

DEFMAP_LIST = [os.path.abspath(p) for p in args.defmap]
OUT_DIR_BASE = (os.path.abspath(args.out_dir)
                if args.out_dir else os.path.dirname(DEFMAP_LIST[0]))
os.makedirs(OUT_DIR_BASE, exist_ok=True)

pat  = re.compile(r"frame_(\d+)\.txt$")

def run_analysis(defmap_path):
    LABEL_DIR  = os.path.abspath(args.labels)
    tag = os.path.splitext(os.path.basename(defmap_path))[0]

    m_layer = re.search(r'_Z(\d+)', tag)
    z_tag = f"Z{m_layer.group(1)}" if m_layer else tag
    filter_tag = "all"

    OUT_DIR = OUT_DIR_BASE
    os.makedirs(OUT_DIR, exist_ok=True)
    LAGS       = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
    CONF_MIN   = args.conf_min
    USE_ABS      = True      # True → |div| ; False → signed value

    CROP_OFFS  = (0, 0)

    defmaps = np.load(defmap_path)
    N, H, W = defmaps.shape
    print("Stack:", defmaps.shape)
    print("Labels:", LABEL_DIR)

    rows = []
    rows_corr = []

    for scale in [1.0, 1.5, 2.0]:
        for txt in sorted(glob.glob(os.path.join(LABEL_DIR, "frame_*.txt"))):
            m = pat.search(os.path.basename(txt))
            if not m:
                continue
            t = int(m.group(1))
            df = pd.read_csv(txt, sep=" ", header=None,
                             names=["cls", "x_c", "y_c", "w", "h", "conf"])
            for _, r in df.iterrows():
                x = int(r.x_c * W) - CROP_OFFS[0]
                y = int(r.y_c * H) - CROP_OFFS[1]
                if not (0 <= x < W and 0 <= y < H):
                    continue

                bb_w = int(r.w * W)
                bb_h = int(r.h * H)
                roi_r = int(max(bb_w, bb_h) * scale)

                for lag in LAGS:
                    t_def = t + lag
                    if t_def < 1 or t_def > N:
                        continue
                    y0 = max(0, y - roi_r)
                    y1 = min(H, y + roi_r + 1)
                    x0 = max(0, x - roi_r)
                    x1 = min(W, x + roi_r + 1)
                    div_patch = defmaps[t_def-1, y0:y1, x0:x1]

                    h_patch, w_patch = div_patch.shape
                    y_grid, x_grid = np.mgrid[0:h_patch, 0:w_patch]
                    cy, cx = (h_patch - 1)/2, (w_patch - 1)/2
                    sigma = roi_r / 1.5  # slightly sharper weighting toward the center
                    kernel = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
                    kernel /= kernel.sum()

                    div_val = float((div_patch * kernel).sum())

                    if USE_ABS:
                        div_val = abs(div_val)

                    if r.conf <= CONF_MIN:
                        continue
                    rows.append(dict(frame=t,
                                     lag=lag,
                                     conf=r.conf,
                                     diverg=div_val,
                                     scale=scale))

    tbl = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR,
                            f"lag_corr_table_{z_tag}_{filter_tag}.csv")
    tbl.to_csv(csv_path, index=False)
    print("→ Tabelle:", csv_path, "  (#rows =", len(tbl), ")")

    for lag in LAGS:
        sub = tbl[tbl.lag == lag]
        if len(sub) < 2:
            print(f"lag {lag}: zu wenig Punkte")
            continue
        r, p = pearsonr(sub["diverg"], sub["conf"])
        label = "Pearson"

        rows_corr.append(dict(lag=lag, r=r, p=p, method=label))

    corr_df = pd.DataFrame(rows_corr).set_index("lag").reindex(LAGS)

    fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
    for scale, sub_df in tbl.groupby("scale"):
        avg_curve = sub_df.groupby("lag")["diverg"].mean()
        sem_curve = sub_df.groupby("lag")["diverg"].sem()
        ax_curve.errorbar(avg_curve.index,
                          avg_curve.values,
                          yerr=sem_curve.values,
                          fmt='-o',
                          label=f"ROI {scale}×",
                          capsize=3)

    ax_curve.legend(loc="upper left")
    n_events = tbl["frame"].nunique()
    min_p = corr_df["p"].min()
    p_str = f"{min_p:.1e}" if min_p < 0.001 else f"{min_p:.3f}"
    ax_curve.text(0.98, 0.95,
                  f"N = {n_events}\np = {p_str}",
                  transform=ax_curve.transAxes,
                  ha="right", va="top",
                  fontsize=10,
                  bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    metric_label = "Divergenz" if "div" in defmap_path else "Magnitude"
    ax_curve.set_title(f"Gemittelte {metric_label} über Lag ({z_tag})")
    ax_curve.set_xlabel("Lag [Frames]")
    ax_curve.set_ylabel(f"Gemittelte {metric_label}")
    ax_curve.set_xticks(LAGS)
    fig_curve.tight_layout(pad=1.0)

    curve_path = os.path.join(OUT_DIR, f"lag_mag_curve_{z_tag}_{filter_tag}.png")
    fig_curve.savefig(curve_path, dpi=300)
    plt.close(fig_curve)
    print("Mittlere Divergenz-Kurve gespeichert:", curve_path)

    # Mittelwert über alle ROIs
    fig_avg, ax_avg = plt.subplots(figsize=(7, 4))
    avg_curve_all = tbl.groupby("lag")["diverg"].mean()
    sem_curve_all = tbl.groupby("lag")["diverg"].sem()
    ax_avg.errorbar(avg_curve_all.index,
                    avg_curve_all.values,
                    yerr=sem_curve_all.values,
                    fmt='-o',
                    color="black",
                    capsize=3,
                    label="ROI gemittelt")
    ax_avg.legend(loc="upper left")

    ax_avg.set_title(f"Gemittelte {metric_label} über alle ROIs ({z_tag})")
    ax_avg.set_xlabel("Lag [Frames]")
    ax_avg.set_ylabel(f"Gemittelte {metric_label}")
    ax_avg.set_xticks(LAGS)

    ax_avg.text(0.98, 0.95,
                f"N = {n_events}\np = {p_str}",
                transform=ax_avg.transAxes,
                ha="right", va="top",
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    fig_avg.tight_layout(pad=1.0)
    curve_avg_path = os.path.join(OUT_DIR, f"lag_mag_curve_avg_{z_tag}_{filter_tag}.png")
    fig_avg.savefig(curve_avg_path, dpi=300)
    plt.close(fig_avg)
    print("Mittlere Magnitude-Kurve (alle ROIs gemittelt) gespeichert:", curve_avg_path)

    stats_csv = os.path.join(OUT_DIR,
                             f"lag_corr_stats_{z_tag}_{filter_tag}.csv")
    corr_df.to_csv(stats_csv)
    print("Statistik‑Tabelle gespeichert:", stats_csv)

    fig_sig, ax_sig = plt.subplots(figsize=(8, 2.6))
    bars = ax_sig.bar(corr_df.index,
                      corr_df["r"],
                      color=["#d73027" if r > 0 else "#4575b4"
                             for r in corr_df["r"]])
    for bar, p_val in zip(bars, corr_df["p"]):
        if p_val < 0.05:
            ax_sig.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "★",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold")
    ax_sig.set_xlabel("Lag [Frames]")
    ax_sig.set_ylabel("Korrelationskoeffizient r")
    ax_sig.set_xticks(LAGS)
    ax_sig.set_title(f"Conf ↔ Magnitude – signifikante Lags (★ p<0.05)")
    ax_sig.axhline(0, lw=1, color="k")
    fig_sig.tight_layout(pad=1.0)
    sig_png = os.path.join(OUT_DIR, f"lag_corr_significant_mag_{z_tag}_{filter_tag}.png")
    fig_sig.savefig(sig_png, dpi=300)
    plt.close(fig_sig)
    print("Signifikanz‑Plot gespeichert:", sig_png)

if __name__ == "__main__":

    for defmap_path in DEFMAP_LIST:
        run_analysis(defmap_path)