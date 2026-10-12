"""Build docs/excavator.stl (all parts assembled at the zero pose) and docs/excavator.png.

Plain Python with numpy, matplotlib and Pillow; Isaac Sim is not needed. Run from the repo root:
    python3 scripts/make_preview.py
Re-run after the meshes or Excavator_Final.urdf change.
"""
import os
import struct
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
urdf = os.path.join(REPO, "Excavator_Final_description", "urdf", "Excavator_Final.urdf")
out_dir = os.path.join(REPO, "docs")
os.makedirs(out_dir, exist_ok=True)
root = ET.parse(urdf).getroot()
mesh_dir = os.path.join(os.path.dirname(urdf), "..", "meshes")

def xyz(el): return np.array([float(v) for v in el.get("xyz").split()])
for el in root.iter("origin"):
    assert all(float(v) == 0 for v in el.get("rpy").split()), "non-zero rpy not handled"

frame = {"base_link": np.zeros(3)}
joints = list(root.iter("joint"))
while len(frame) < len(joints) + 1:
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        if p in frame and c not in frame:
            frame[c] = frame[p] + xyz(j.find("origin"))

def read_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    return rec["v"].astype(np.float64)

parts = []
for link in root.iter("link"):
    vis = link.find("visual")
    mesh = vis.find("geometry/mesh")
    scale = np.array([float(s) for s in mesh.get("scale").split()])
    tris = read_stl(os.path.join(mesh_dir, os.path.basename(mesh.get("filename")))) * scale
    tris += frame[link.get("name")] + xyz(vis.find("origin"))
    parts.append((link.get("name"), tris))
    print(link.get("name"), len(tris), "triangles")

allt = np.concatenate([t for _, t in parts]).astype("<f4")
normals = np.cross(allt[:, 1] - allt[:, 0], allt[:, 2] - allt[:, 0])
normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
with open(os.path.join(out_dir, "excavator.stl"), "wb") as f:
    f.write(b"Excavator_Final assembled at zero joint pose".ljust(80, b" "))
    f.write(struct.pack("<I", len(allt)))
    rec = np.zeros(len(allt), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = normals, allt
    f.write(rec.tobytes())

colors = ["#8a8f98", "#e0a526", "#d4862a", "#c4672b", "#5b6470"]
fig = plt.figure(figsize=(8, 7), dpi=130)
ax = fig.add_subplot(111, projection="3d")
light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
for (name, tris), col in zip(parts, colors):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.45 + 0.55 * np.abs(n @ light)
    base = np.array(matplotlib.colors.to_rgb(col))
    pc = Poly3DCollection(tris, facecolors=np.clip(base * shade[:, None], 0, 1), edgecolors="none")
    ax.add_collection3d(pc)
pts = allt.reshape(-1, 3)
lo, hi = pts.min(0), pts.max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2
ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-35); ax.set_axis_off()
plt.tight_layout()
png = os.path.join(out_dir, "excavator.png")
plt.savefig(png, facecolor="white")
# Trim the empty margin matplotlib leaves around 3D axes.
im = Image.open(png).convert("RGB")
ys, xs = np.where((np.asarray(im) < 245).any(2))
pad = 30
im.crop((xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)).save(png, optimize=True)
print("bbox (m):", lo.round(3), hi.round(3))
