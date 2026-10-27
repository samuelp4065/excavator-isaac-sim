"""Find excavator poses for motion planning (plain Python, no Isaac Sim needed).

    python3 scripts/getpose.py ik X Y Z PITCH [--save NAME]   # bucket edge -> joint angles
    python3 scripts/getpose.py fk SWING ARM POLE BUCKET [--deg] [--save NAME]
    python3 scripts/getpose.py list                           # saved poses (poses.json)
    python3 scripts/getpose.py clear                          # start over (old file kept as a backup)
    python3 scripts/getpose.py delete NAME [NAME ...]
    python3 scripts/getpose.py rename OLD NEW
    python3 scripts/getpose.py files                          # pose files in scripts/
Add --file NAME before the command to use scripts/NAME.json instead of
poses.json, e.g. one file per path:  getpose.py --file dig ik 0.08 -0.48 0 2.5 --save start
    python3 scripts/getpose.py pile                           # where the rocks/pit/bin are

Positions are in metres, world frame (z up, ground at 0); angles in radians
unless --deg. X Y Z is the middle of the bucket's cutting edge. PITCH is the
bucket angle in the boom plane (kinematics.py: arm - pole - bucket):
    2.5  cutting edge pointing down and back at the machine (digging)
    0.5  bucket opening straight up (carrying)
   -0.2  tilted back further (holds a load better; the bucket's range may stop it)
    ~3.5 tipped forward (dumping)

Every pose is checked against the collision-free joint ranges
(kinematics.JOINT_LIMITS, beyond them a link passes through its neighbour) and
against the dirt (whole bucket mesh vs the ground / pit in rocks.py).

--save NAME stores the pose in scripts/poses.json (or the --file); the live
sim's G key adds the robot's current pose to POSE_FILE in live_controller.py, so you can jog it
with the keyboard, record waypoints, and build a path from them.

Joint lists are printed in the order (swing, arm, pole, bucket) =
kinematics.JOINTS. In live_controller.py set them by name, e.g.
ctx.targets[ctx.idx["arm_joint"]] = ..., since the sim's DOF order may differ.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as K
import pick_motion as M
import rocks as R

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_PATH = os.path.join(SCRIPT_DIR, "poses.json")  # --file picks another, e.g. dig.json


def pose_file(name):
    """Pose file path: a bare name like 'dig' or 'dig.json' lives in scripts/."""
    if not name.endswith(".json"):
        name += ".json"
    return name if os.path.dirname(name) else os.path.join(SCRIPT_DIR, name)


def check(q):
    """Problems with a joint pose (empty list = fine)."""
    problems = []
    for n, (lo, hi) in K.JOINT_LIMITS.items():
        if not lo <= q[n] <= hi:
            problems.append(f"{n} {np.degrees(q[n]):.0f} deg is outside its collision-free range "
                            f"{np.degrees(lo):.0f}..{np.degrees(hi):.0f} deg")
    clear = M.bucket_clearance(q, R.terrain_height)
    if clear < 0:
        problems.append(f"bucket is {-clear * 1000:.0f} mm inside the dirt")
    return problems


def show(q, title=""):
    q = K.as_dict(q)
    tip, pitch = K.fk(q)
    pitch = K._near(pitch, 0.0)
    clear = M.bucket_clearance(q, R.terrain_height)
    reach = np.linalg.norm(tip[:2] - K.SWING_ORIGIN[:2])
    print(title)
    print(f"  joints (rad): " + "  ".join(f"{n.removesuffix('_joint')}={q[n]:+.4f}" for n in K.JOINTS))
    print(f"  joints (deg): " + "  ".join(f"{n.removesuffix('_joint')}={np.degrees(q[n]):+.1f}" for n in K.JOINTS))
    print(f"  as list     : [{', '.join(f'{q[n]:.4f}' for n in K.JOINTS)}]")
    print(f"  cutting edge: x={tip[0]:+.4f} y={tip[1]:+.4f} z={tip[2]:+.4f} m, pitch {pitch:+.3f} rad "
          f"({np.degrees(pitch):+.0f} deg), reach {reach:.3f} m from the swing axis")
    print(f"  bucket clearance above dirt: {clear * 1000:+.0f} mm")
    problems = check(q)
    print("  OK" if not problems else "  PROBLEMS:\n    " + "\n    ".join(problems))
    return q, tip, pitch


def save(name, q, source, **extra):
    poses = load()
    tip, pitch = K.fk(q)
    poses[name] = dict(joints={n: round(float(q[n]), 5) for n in K.JOINTS},
                       tip=[round(float(v), 5) for v in tip], pitch=round(float(K._near(pitch, 0.0)), 5),
                       source=source, **extra)
    write(poses)
    print(f"saved '{name}' to {POSES_PATH}")


def next_live_name():
    """First unused live_N (the G key's names), so nothing gets overwritten."""
    poses, i = load(), 1
    while f"live_{i}" in poses:
        i += 1
    return f"live_{i}"


def write(poses):
    with open(POSES_PATH, "w") as f:
        json.dump(poses, f, indent=2)


def load():
    try:
        with open(POSES_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def cmd_ik(a):
    tip = np.array([a.x, a.y, a.z])
    try:
        q = M.ik_fwd(tip, a.pitch)
    except ValueError as e:
        print(f"no pose: {e}")
        # Say what would work: nearest reachable heights and pitches here.
        hints = []
        for pitch in np.arange(-0.5, 3.6, 0.25):
            for z in np.arange(-0.1, 0.6, 0.01):
                try:
                    M.ik_fwd(np.array([a.x, a.y, z]), pitch)
                except ValueError:
                    continue
                hints.append((pitch, z))
        if hints:
            ps = sorted({round(p, 2) for p, _ in hints})
            print(f"  at x={a.x} y={a.y}, reachable pitches: {ps[0]:+.2f} .. {ps[-1]:+.2f} rad")
            zs = [z for p, z in hints if abs(p - a.pitch) < 0.13]
            if zs:
                print(f"  at pitch {a.pitch:+.2f}: reachable z {min(zs):+.2f} .. {max(zs):+.2f} m")
        else:
            print(f"  x={a.x} y={a.y} is out of reach at any height/pitch "
                  f"(forward reach ~0.25-0.55 m near the ground, measured from the swing axis)")
        return 1
    q, _, _ = show(q, f"pose for edge ({a.x}, {a.y}, {a.z}), pitch {a.pitch}:")
    if a.save:
        save(a.save, q, "getpose ik")
    return 0


def cmd_fk(a):
    vals = np.radians(a.q) if a.deg else np.array(a.q)
    q = dict(zip(K.JOINTS, vals))
    show(q, "pose:")
    if a.save:
        save(a.save, q, "getpose fk")
    return 0


def cmd_list(a):
    poses = load()
    if not poses:
        print(f"no saved poses ({POSES_PATH})")
        return 0
    for name, p in poses.items():
        j = p["joints"]
        print(f"{name:16s} [{', '.join(f'{j[n]:+.4f}' for n in K.JOINTS)}]  edge "
              f"({p['tip'][0]:+.3f}, {p['tip'][1]:+.3f}, {p['tip'][2]:+.3f}) pitch {p['pitch']:+.2f}"
              f"  [{p.get('source', '')}]")
        if "rocks_top" in p:
            print(f"{'':16s} top rocks then: " + "  ".join(f"({x:+.3f}, {y:+.3f}, {z:+.3f})" for x, y, z in p["rocks_top"]))
    return 0


def cmd_clear(a):
    if not os.path.exists(POSES_PATH):
        print("nothing to clear")
        return 0
    backup = POSES_PATH.replace(".json", time.strftime("_%Y%m%d_%H%M%S.json"))
    os.rename(POSES_PATH, backup)
    print(f"cleared; old poses kept in {backup} (the next G saves live_1)")
    return 0


def cmd_delete(a):
    poses = load()
    missing = [n for n in a.names if n not in poses]
    for n in a.names:
        poses.pop(n, None)
    write(poses)
    print(f"deleted {[n for n in a.names if n not in missing]}" + (f"; not found: {missing}" if missing else ""))
    return 1 if missing else 0


def cmd_rename(a):
    poses = load()
    if a.old not in poses:
        print(f"no pose '{a.old}'")
        return 1
    if a.new in poses:
        print(f"'{a.new}' already exists; delete it first")
        return 1
    poses = {(a.new if k == a.old else k): v for k, v in poses.items()}  # keep the order
    write(poses)
    print(f"renamed '{a.old}' -> '{a.new}' (update PATH in live_controller.py if it uses the old name)")
    return 0


def cmd_files(a):
    found = False
    for f in sorted(os.listdir(SCRIPT_DIR)):
        if not f.endswith(".json"):
            continue
        try:
            poses = json.load(open(os.path.join(SCRIPT_DIR, f)))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(poses, dict) and all(isinstance(v, dict) and "joints" in v for v in poses.values()):
            found = True
            print(f"{f:32s} {len(poses):3d} poses: {', '.join(poses)}")
    if not found:
        print("no pose files in scripts/")
    return 0


def cmd_pile(a):
    centre = R.SWING_AXIS_XY + R.PILE_DISTANCE * R.BOOM_DIR_XY
    bin_dir = R._rotate_xy(R.BOOM_DIR_XY, R.BIN_SWING_DEG)
    bin_c = R.SWING_AXIS_XY + R.BIN_DISTANCE * bin_dir
    print(f"pile style '{R.PILE_STYLE}', centre x={centre[0]:+.3f} y={centre[1]:+.3f} "
          f"({R.PILE_DISTANCE} m from the swing axis at swing 0)")
    if R.PILE_STYLE == "pit":
        print(f"  pit: rim radius {R.PIT_RIM_RADIUS} m at z=0, floor radius {R.PIT_FLOOR_RADIUS} m at "
              f"z={-R.PIT_DEPTH}; settled rocks fill it to about z=+0.01")
    if R.DUMP_BIN:
        print(f"dump bin centre x={bin_c[0]:+.3f} y={bin_c[1]:+.3f} (swing {R.BIN_SWING_DEG:+.0f} deg), "
              f"inside {R.BIN_SIZE} m wide, walls {R.BIN_WALL_HEIGHT} m high")
    print("live rock positions: press G in the sim (saved with the pose, see 'list')")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="poses.json", help="pose file in scripts/ (default poses.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("files", help="list pose files in scripts/")
    p = sub.add_parser("ik", help="cutting edge xyz + pitch -> joint angles")
    for n in ("x", "y", "z", "pitch"):
        p.add_argument(n, type=float)
    p.add_argument("--save", metavar="NAME")
    p = sub.add_parser("fk", help="joint angles -> cutting edge xyz + pitch")
    p.add_argument("q", type=float, nargs=4, metavar=("SWING", "ARM", "POLE", "BUCKET"))
    p.add_argument("--deg", action="store_true", help="angles in degrees")
    p.add_argument("--save", metavar="NAME")
    sub.add_parser("list", help="saved poses")
    sub.add_parser("pile", help="pile, pit and bin positions")
    sub.add_parser("clear", help="start over (old poses kept as a backup file)")
    p = sub.add_parser("delete", help="delete saved poses")
    p.add_argument("names", nargs="+")
    p = sub.add_parser("rename", help="rename a saved pose")
    p.add_argument("old")
    p.add_argument("new")
    a = ap.parse_args()
    global POSES_PATH
    POSES_PATH = pose_file(a.file)
    return {"ik": cmd_ik, "fk": cmd_fk, "list": cmd_list, "pile": cmd_pile, "clear": cmd_clear,
            "delete": cmd_delete, "rename": cmd_rename, "files": cmd_files}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
