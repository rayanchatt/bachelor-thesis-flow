"""
RGB-Stapelbuilder für den 8/6-er-Datensatz
– erwartet pro Frame genau drei Dateien …_t####_z0001/2/3.png
– erzeugt RGB-Frames im Unterordner rgb/ als frame_###.png
"""

import os, re, cv2
from collections import defaultdict

input_dir  = "/Users/macbook/Downloads/division_sequence_8"
output_dir = os.path.join(input_dir, "rgb")
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r".+?_t(\d+)_z(\d+)\.(png|tif|jpg|jpeg)$", re.I)
groups  = defaultdict(dict)

for fname in os.listdir(input_dir):
    m = pattern.match(fname)
    if not m:
        continue
    frame = int(m.group(1))
    z     = int(m.group(2))         
    groups[frame][z] = os.path.join(input_dir, fname)


written = 0
for frame_idx in sorted(groups):
    files = groups[frame_idx]
    if not all(z in files for z in (1, 2, 3)):
        print(f"Frame {frame_idx:03d}: z-Kanäle fehlen, übersprungen")
        continue

    # Graustufenbilder laden
    z1 = cv2.imread(files[1], cv2.IMREAD_GRAYSCALE)
    z2 = cv2.imread(files[2], cv2.IMREAD_GRAYSCALE)
    z3 = cv2.imread(files[3], cv2.IMREAD_GRAYSCALE)

    def to_u8(img):
        return img if img.dtype == "uint8" else cv2.convertScaleAbs(img)
    z1, z2, z3 = map(to_u8, (z1, z2, z3))

    # Zu RGB mergen (R=Z3, G=Z2, B=Z1)
    rgb = cv2.merge([z1, z2, z3])

    out_name = f"frame_{frame_idx:03d}.png"
    cv2.imwrite(os.path.join(output_dir, out_name), rgb)
    written += 1
    print(f"geschrieben: {out_name}")

print(f"\n Fertig – {written} RGB-Frames unter {output_dir}")