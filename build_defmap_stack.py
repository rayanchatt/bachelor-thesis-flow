import os, glob, argparse, cv2, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# CLI / Argumente
p = argparse.ArgumentParser(
        description="Erzeuge DefMap‑Stack aus RGB‑Frames via Farnebäck‑Flow")
import re
p.add_argument("--rgb_dir", required=True,
               help="Ordner mit Rohdaten-Bildern '..._tXXXX_zXXXX.tif'.")
p.add_argument("--out_dir", required=True,
               help="Zielordner für PNG/Videos und *.npy‑Stack.")
p.add_argument("--metric", choices=["div", "mag"], default="div",
               help="'div' = Divergenz ∇·v  |  'mag' = Betrag ‖v‖  (Default: div)")
p.add_argument("--sigma", type=float, default=1.5,
               help="Gauss‑Sigma zur Flow‑Glättung (0 = aus).")
p.add_argument("--channel", type=int, choices=[0,1,2],
               help="Nur diesen Farbkanal verarbeiten (0=B/Z1, 1=G/Z2, 2=R/Z3). "
                    "Ohne Angabe → Graustufen‑Mix.")
p.add_argument("--crop", type=int, nargs=4, metavar=("x0","y0","w","h"),
               help="Crop‑Box in Pixeln (x0 y0 w h).")
p.add_argument("--save_png", action="store_true",
               help="PNG‑Preview der einzelnen DefMaps speichern.")
p.add_argument("--save_video", action="store_true",
               help="MP4‑Preview schreiben (requires --save_png).")
p.add_argument("--fps", type=int, default=8,
               help="FPS des Videos (Default 8).")
args = p.parse_args()

RGB_DIR   = os.path.abspath(args.rgb_dir)
OUT_DIR   = os.path.abspath(args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

# Farnebäck‑Parameter
FLOW_KW = dict(pyr_scale=0.5, levels=3, winsize=15,
               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

CMAP      = "coolwarm"
VMAX_DIV  = 4.0          # Clip‑Grenze für Divergenz‑PNG
VMAX_MAG  = 4.0          # Clip für Betrag

# Frame‑Liste
# Lade Rohdaten aus Ordner, sortiert nach Zeit und Z-Ebene
RAW_PATTERN = re.compile(r".*?_t(\d+)_z(\d+).*?\.png$")
all_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
frame_dict = {}
for f in all_files:
    m = RAW_PATTERN.match(os.path.basename(f))
    if not m:
        continue
    t, z = int(m.group(1)), int(m.group(2))
    frame_dict.setdefault(z, {})[t] = f

if not frame_dict:
    raise SystemExit("Keine Rohdaten im erwarteten Format gefunden.")

z_slices = sorted(frame_dict.keys())
t_list = sorted(next(iter(frame_dict.values())).keys())
print(f"➕ Rohdaten geladen: Z-Slices = {z_slices}, Zeitschritte = {t_list}")

vw = None
# Nur eine z-Ebene verwenden für Flow
all_stacks = {}
for Z_USE in z_slices:
    frames = [frame_dict[Z_USE][t] for t in t_list]
    stack = []

    for i in tqdm(range(len(frames)-1), desc=f"Z{Z_USE}"):
        if args.channel is None:
            g1 = cv2.imread(frames[i],   cv2.IMREAD_GRAYSCALE)
            g2 = cv2.imread(frames[i+1], cv2.IMREAD_GRAYSCALE)
        else:
            g1 = cv2.split(cv2.imread(frames[i]))[args.channel]
            g2 = cv2.split(cv2.imread(frames[i+1]))[args.channel]

        if args.crop:
            x0,y0,w,h = args.crop
            g1 = g1[y0:y0+h, x0:x0+w]
            g2 = g2[y0:y0+h, x0:x0+w]

        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **FLOW_KW)
        vx, vy = flow[...,0], flow[...,1]

        if args.sigma > 0:
            vx = cv2.GaussianBlur(vx, (0,0), sigmaX=args.sigma)
            vy = cv2.GaussianBlur(vy, (0,0), sigmaX=args.sigma)

        if args.metric == "mag":
            data = np.sqrt(vx**2 + vy**2).astype(np.float32)
            clip_lo, clip_hi = 0.0, VMAX_MAG
        else:
            dvx_dx = np.gradient(vx, axis=1)
            dvy_dy = np.gradient(vy, axis=0)
            data = (dvx_dx + dvy_dy).astype(np.float32)
            clip_lo, clip_hi = -VMAX_DIV, VMAX_DIV

        stack.append(data)

        if args.save_png or args.save_video:
            shown = np.clip(data, clip_lo, clip_hi)
            plt.figure(figsize=(4,3))
            plt.imshow(shown, cmap=CMAP, vmin=clip_lo, vmax=clip_hi)
            plt.axis("off")
            plt.tight_layout(pad=0)
            png_path = os.path.join(OUT_DIR, f"defmap_Z{Z_USE}_{i+1:03d}.png")
            if args.save_png:
                plt.savefig(png_path, dpi=120, bbox_inches="tight", pad_inches=0)
            plt.close()
            if vw is not None:
                vw.write(cv2.imread(png_path))

    stack = np.stack(stack)
    base_npy = f"defmap_stack_Z{Z_USE}_{args.metric}.npy"
    out_npy = os.path.join(OUT_DIR, base_npy)
    np.save(out_npy, stack)
    print(f"DefMap‑Stack Z{Z_USE} → {out_npy}  {stack.shape}")

if vw is not None:
    vw.release()
    print(f"Preview‑Video → {vw_path}")
