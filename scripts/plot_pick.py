"""Side-view plots of pick_rock.py trials (system Python: numpy + matplotlib).

    python3 scripts/plot_pick.py [strategy ...]

For each docs/pick_rock/<strategy>.npz writes <strategy>.png: left, the bucket
mesh in the boom plane at a few moments (from the commanded joint angles) with
the target rock's path; right, target rock height and bucket-tip height over
time, coloured by phase.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as K

OUT_DIR = os.path.join(K.REPO, "docs", "pick_rock")
DOF = ("swing_joint", "arm_joint", "pole_joint", "bucket_joint")  # articulation order


def _bucket_triangles():
    import struct
    data = open(K.BUCKET_STL, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)
    v = rec["v"].reshape(-1, 3).astype(float) * 0.001
    return np.array([K._to_plane(p)[0] for p in v]).reshape(-1, 3, 2)


BUCKET_TRIS = _bucket_triangles()  # (n, 3, 2): mesh triangles projected onto the boom plane


def bucket_triangles(q):
    pts, pitch = K.fk_plane(q["arm_joint"], q["pole_joint"], q["bucket_joint"])
    return (BUCKET_TRIS - K.BUCKET_PIN) @ K._rot(pitch).T + pts["bucket_pin"]


# Dirt profile along the boom through the pile centre, mirroring rocks.py
# (which needs pxr): (radius, height) points, pile centre ~0.45 m out.
PROFILES = {
    "crater": [(0.0, 0.0), (0.05, 0.0), (0.15, 0.06), (0.22, 0.0), (0.35, 0.0)],
    "pit": [(0.0, -0.06), (0.06, -0.06), (0.12, 0.0), (0.35, 0.0)],
}
PILE_U = 0.45


def scene_of(name, d):
    if "scene" in d:
        return str(d["scene"])
    return "pit" if name.startswith(("pit", "final")) else "crater"


def to_plane(xyz, swing):
    """World point -> (u, z) in the boom plane at this swing."""
    d = xyz[:2] - K.SWING_ORIGIN[:2]
    c, s = np.cos(-swing), np.sin(-swing)
    h = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    return np.array([h @ K.U_HAT[:2], xyz[2]])


def plot(name, ax_side=None, ax_time=None, title=True):
    d = np.load(os.path.join(OUT_DIR, f"{name}.npz"), allow_pickle=True)
    t, phase, qc = d["t"], d["phase"], d["q_cmd"]
    own = ax_side is None
    if own:
        fig, (ax_side, ax_time) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    phases = list(dict.fromkeys(phase))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(phases)))
    # Side view: bucket at the end of each phase.
    for p, c in zip(phases, colors):
        i = np.where(phase == p)[0][-1]
        q = dict(zip(DOF, qc[i]))
        ax_side.add_collection(PolyCollection(bucket_triangles(q), facecolor=c, edgecolor="none", alpha=0.25))
        ax_side.plot(*to_plane(d["tip"][i], q["swing_joint"]), "x", color=c, ms=7, label=f"end of {p}")
    swing = qc[:, 0]
    rock = np.array([to_plane(x, s) for x, s in zip(d["target_pos"], swing)])
    tip = np.array([to_plane(x, s) for x, s in zip(d["tip"], swing)])
    ax_side.plot(tip[:, 0], tip[:, 1], "-", color="0.5", lw=0.8, label="cutting edge path")
    ax_side.plot(rock[:, 0], rock[:, 1], "-", color="crimson", lw=2, label="target rock path")
    ax_side.plot(*rock[0], "o", color="crimson", ms=8, mfc="none")
    ax_side.plot(*rock[-1], "o", color="crimson", ms=8)
    prof = np.array(PROFILES[scene_of(name, d)])
    r = np.concatenate([-prof[::-1, 0], prof[:, 0]])
    ax_side.fill_between(PILE_U + r, np.concatenate([prof[::-1, 1], prof[:, 1]]), -0.1,
                         color="saddlebrown", alpha=0.25, lw=0)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("reach u (m)")
    ax_side.set_ylabel("height z (m)")
    ax_side.set_xlim(0.15, 0.62)
    ax_side.set_ylim(-0.08, 0.62)
    ax_side.legend(fontsize=6.5, loc="upper left")
    # Time view.
    for p, c in zip(phases, colors):
        m = phase == p
        ax_time.axvspan(t[m][0], t[m][-1], color=c, alpha=0.12)
        ax_time.text((t[m][0] + t[m][-1]) / 2, 0.58, p, ha="center", fontsize=7, rotation=90, va="top")
    ax_time.plot(t, d["target_pos"][:, 2], color="crimson", lw=2, label="target rock z")
    ax_time.plot(t, d["tip"][:, 2], color="0.4", lw=1, label="cutting edge z")
    ax_time.axhline(0.10, color="green", ls="--", lw=1, label="success height")
    ax_time.set_ylim(-0.08, 0.6)
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("z (m)")
    ax_time.legend(fontsize=7, loc="upper right")
    if title:
        ax_side.set_title(f"{name}: side view", fontsize=10)
        ax_time.set_title("heights over time", fontsize=10)
    if own:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(OUT_DIR) if f.endswith(".npz"))
    for n in names:
        plot(n)
        print("wrote", os.path.join(OUT_DIR, f"{n}.png"))
