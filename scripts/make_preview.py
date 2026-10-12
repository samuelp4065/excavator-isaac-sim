"""Render docs/excavator.png (the README picture) from the CAD export Excavator_Final.stl.

Plain Python with numpy, matplotlib and Pillow; Isaac Sim is not needed. Run from anywhere:
    python3 scripts/make_preview.py
Re-run after replacing Excavator_Final.stl.
"""
import os
import struct

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STL_PATH = os.path.join(REPO, "Excavator_Final.stl")
PNG_PATH = os.path.join(REPO, "docs", "excavator.png")
COLOR = "#e0a526"


def read_stl(path):
    """Triangles of a binary STL as an (n, 3, 3) array."""
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.frombuffer(
        data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        count=n, offset=84,
    )
    return rec["v"].astype(np.float64)


tris = read_stl(STL_PATH)
normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
light = np.array([0.4, -0.6, 0.7])
light /= np.linalg.norm(light)
shade = 0.4 + 0.6 * np.abs(normals @ light)
colors = np.clip(np.array(matplotlib.colors.to_rgb(COLOR)) * shade[:, None], 0, 1)

fig = plt.figure(figsize=(8, 7), dpi=130)
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolors="none"))
pts = tris.reshape(-1, 3)
lo, hi = pts.min(0), pts.max(0)
c, r = (lo + hi) / 2, (hi - lo).max() / 2
ax.set_xlim(c[0] - r, c[0] + r)
ax.set_ylim(c[1] - r, c[1] + r)
ax.set_zlim(c[2] - r, c[2] + r)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=-35)
ax.set_axis_off()
plt.tight_layout()
os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
plt.savefig(PNG_PATH, facecolor="white")

# Trim the empty margin matplotlib leaves around 3D axes.
im = Image.open(PNG_PATH).convert("RGB")
ys, xs = np.where((np.asarray(im) < 245).any(2))
pad = 30
im.crop((xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)).save(PNG_PATH, optimize=True)
print(f"Wrote {PNG_PATH} ({len(tris)} triangles)")
