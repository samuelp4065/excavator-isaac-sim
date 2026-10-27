"""Excavator control logic, hot-reloaded by run_live.py.

Edit and save this file while Isaac Sim is running: run_live.py notices the change
and reloads it without restarting the sim. Joint targets carry over, so the robot
doesn't snap back. If the file has an error, the traceback is printed and the
previous working version keeps running until you save a fixed one.

`step(ctx)` is called every physics step. Useful fields on ctx:
    ctx.t, ctx.dt      sim time since start / physics step (s)
    ctx.step           physics step counter
    ctx.targets        commanded joint positions (rad); modify in place
    ctx.positions      measured joint positions (rad), read-only snapshot
    ctx.idx[name]      DOF index of a joint, e.g. ctx.idx["arm_joint"]
    ctx.held           set of carb.input.KeyboardInput currently held down
    ctx.excavator      the SingleArticulation, for anything else
    ctx.rocks          the RockSet: .reset(), .positions(), .count
"""
import json
import os

import carb.input
import numpy as np

import getpose
import kinematics
import pick_motion  # imported once per session (as are getpose/kinematics): edits need a run_live.py restart

MODE = "keyboard"  # "keyboard" or "sweep"

# Drive gains, re-applied on every reload.
KP = 1e4
KD = 1e3

K = carb.input.KeyboardInput
# All joints are 'continuous' in the URDF, so these clamps (radians) are the only
# bound on motion.
JOINT_KEYS = {
    # name:          (key +,   key -,   lo,     hi)
    "swing_joint":  (K.LEFT,  K.RIGHT, -np.pi, np.pi),
    # Collision-free ranges (kinematics.JOINT_LIMITS): beyond them a link swings
    # through its neighbour, since the sim doesn't collide neighbouring links.
    "arm_joint":    (K.Q,     K.A,     np.radians(-50), np.radians(110)),
    "pole_joint":   (K.W,     K.S,     np.radians(-5),  np.radians(130)),
    "bucket_joint": (K.E,     K.D,     np.radians(-8),  np.radians(278)),  # hits the stick at -80..-10 deg
}
RESET_KEY = K.R  # move back to HOME
ROCKS_RESET_KEY = K.T

# Video cameras for V (fixed in the world; eye = camera position, target = what it
# looks at, focal = lens in mm: smaller = wider). Positions in metres; the pit
# centre is about (0.06, -0.36, 0), the bin (0.44, 0.11, 0), the swing axis (0, 0.04).
# The cameras appear under /World/RecCams (viewport camera menu: look through
# them). Edit and save to move them; while recording, changes wait until V stops it.
CAMERAS = {
    "overview": dict(eye=(0.95, -0.95, 0.75), target=(0.15, -0.20, 0.10), focal=18),
    "side":     dict(eye=(0.85, -0.10, 0.25), target=(0.03, -0.20, 0.15), focal=18),
    "pit":      dict(eye=(0.32, -0.75, 0.30), target=(0.06, -0.36, 0.00), focal=18),
    "top":      dict(eye=(0.15, -0.15, 1.40), target=(0.15, -0.15, 0.00), focal=18),
}
RECORD_RESOLUTION = (720, 1280)  # per camera; the grid video is 2x2 of these
RECORD_EVERY = 1                # physics steps per frame: 60 Hz / 2 = 30 fps

# Home pose: where R moves the arm, and where run_live.py starts it (the rock
# pile is poured with the arm there, so keep the bucket up out of the pit).
#   None                       all joints at 0 (the CAD pose)
#   "live_1"                   a pose saved in POSE_FILE (G key / getpose.py --save)
#   [swing, arm, pole, bucket] joint angles in radians (getpose.py prints these)
# R picks up changes immediately (hot-reload); the start pose needs a run_live restart.
HOME = HOME = [0.0, 0.3, 0.8, 2.1]
HOME_SECONDS = 2.0  # time for the R move
RECORD_KEY = K.V  # start / stop recording video from CAMERAS (see recorder.py)
PICK_KEY = K.P  # plan and play pick_motion's scoop of the top rock near the pile centre
SAVE_POSE_KEY = K.G  # save the current pose (+ top rock positions) to POSE_FILE, see getpose.py
PATH_KEY = K.Y  # play the poses in POSE_FILE (all in saved order, or PATH)

# Pose file in scripts/ that G saves to and Y reads from. One file per path keeps
# them apart: change this (e.g. "dig.json"), save, and the next G starts live_1 there.
POSE_FILE = "poses.json"

# Y plays every pose in POSE_FILE in the order they were saved, PATH_SECONDS
# per move. For your own order/timing, set PATH to a list of (pose name, seconds
# to get there from the previous pose), e.g.
#     PATH = [("live_1", 3.0), ("live_2", 2.0), ("live_1", 2.0)]
# Each move is a smooth straight line in joint space. Edit and save this file to
# change it (hot-reloaded); POSE_FILE is re-read on every Y press, so newly saved
# poses (G) are included straight away.
PATH = None
PATH_SECONDS = 2.5
# True: flow through the middle poses without stopping (smooth curve per joint
# that never overshoots, so joints stay between neighbouring poses and inside
# their ranges); only the first and last pose are reached at rest.
# False: stop at every pose (each move eased in and out on its own).
PATH_SMOOTH = True
SPEED = 0.5  # rad/s while a key is held (~29 deg/s)

# Used when MODE == "sweep": each joint goes lo -> hi -> lo at freq (rad/s).
JOINT_MOTION = {
    # name:          (lo,    hi,   freq)
    "swing_joint":  (0.0,   np.pi, 0.3),
    "arm_joint":    (0.0,   0.8,   0.5),
    "pole_joint":   (0.0,   0.8,   0.7),  # stay inside the collision-free ranges
    "bucket_joint": (0.0,   1.0,   0.9),  # (negative bucket angles go through the stick)
}

PRINT_EVERY = 6  # steps between angle printouts while moving


def step(ctx):
    record(ctx)
    if MODE == "sweep":
        moved = sweep(ctx)
    else:
        moved = keyboard(ctx)

    if moved and ctx.step % PRINT_EVERY == 0:
        print("  ".join(
            f"{name.removesuffix('_joint')}={np.rad2deg(ctx.positions[ctx.idx[name]]):+7.1f}°"
            f" (cmd {np.rad2deg(ctx.targets[ctx.idx[name]]):+7.1f}°)"
            for name in JOINT_KEYS
        ))


def _load_recorder():
    """Fresh copy of recorder.py (picks up edits; plain import would be cached)."""
    import types
    path = os.path.join(os.path.dirname(kinematics.__file__), "recorder.py")
    mod = types.ModuleType("recorder")
    mod.__file__ = path
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), mod.__dict__)
    return mod, os.path.getmtime(path)


def record(ctx):
    """Keep the CAMERAS in the scene (free until recording: pick them in the
    viewport's camera menu to look through them); V starts / stops recording."""
    rec = getattr(ctx, "recorder", None)
    path = os.path.join(os.path.dirname(kinematics.__file__), "recorder.py")
    stale = rec is not None and not rec.active and (
        rec.cameras != CAMERAS or rec.resolution != tuple(RECORD_RESOLUTION)
        or getattr(ctx, "recorder_mtime", None) != os.path.getmtime(path))
    if rec is None and getattr(ctx, "recorder_error", None) and \
            getattr(ctx, "recorder_mtime", None) == os.path.getmtime(path):
        return  # failed for this version of recorder.py (message printed); retried when it changes
    if rec is None or stale:
        if rec is not None:  # release the old one's render products (older versions have no close())
            for name, ann in list(getattr(rec, "annotators", {}).items()):
                for fn in (ann.detach, rec.products[name].destroy):
                    try:
                        fn()
                    except Exception:
                        pass
        try:
            mod, ctx.recorder_mtime = _load_recorder()
            rec = ctx.recorder = mod.Recorder(CAMERAS, RECORD_RESOLUTION, RECORD_EVERY,
                                              fps=round(1 / (ctx.dt * RECORD_EVERY)))
            ctx.recorder_error = None
        except Exception as e:  # don't stop the controller over the cameras
            print(f"[rec] cameras unavailable: {e}")
            ctx.recorder_error, ctx.recorder = str(e), None
            ctx.recorder_mtime = os.path.getmtime(path)
            return
    if RECORD_KEY in ctx.held:
        ctx.held.discard(RECORD_KEY)
        try:
            rec.stop() if rec.active else rec.start()
        except Exception as e:
            print(f"[rec] {'stop' if rec.active else 'start'} failed: {e}")
            rec.active = False
    rec.step()


def keyboard(ctx):
    if pick(ctx) or path(ctx):
        return True
    if SAVE_POSE_KEY in ctx.held:
        ctx.held.discard(SAVE_POSE_KEY)
        save_pose(ctx)
    moved = False
    if ROCKS_RESET_KEY in ctx.held:
        ctx.held.discard(ROCKS_RESET_KEY)  # once per press, not every step
        ctx.rocks.reset()
    if RESET_KEY in ctx.held:
        ctx.held.discard(RESET_KEY)
        q = home_pose()
        if q is not None:
            start_moves(ctx, [("home", q, HOME_SECONDS)], "home")
            return True
    for name, (key_up, key_down, lo, hi) in JOINT_KEYS.items():
        direction = (key_up in ctx.held) - (key_down in ctx.held)
        if direction:
            i = ctx.idx[name]
            ctx.targets[i] = np.clip(ctx.targets[i] + direction * SPEED * ctx.dt, lo, hi)
            moved = True
    return moved


def home_pose():
    """HOME as {joint: angle}, or None (with a message) if it's not usable."""
    if HOME is None:
        q = dict.fromkeys(kinematics.JOINTS, 0.0)
    elif isinstance(HOME, str):
        poses = load_poses()
        if HOME not in poses:
            print(f"[home] '{HOME}' is not in {POSE_FILE}")
            return None
        q = dict(poses[HOME]["joints"])
    else:
        q = dict(zip(kinematics.JOINTS, (float(v) for v in HOME)))
    if not kinematics.in_limits(q):
        print(f"[home] {HOME} is outside the collision-free joint ranges")
        return None
    return q


def start_moves(ctx, moves, what, smooth=False):
    """Queue joint-space moves [(label, {joint: angle}, seconds), ...] from the
    current targets; path() plays them back. Each move is eased in and out, or
    with smooth=True the whole path is one curve that only stops at the ends."""
    q0 = np.array([ctx.targets[ctx.idx[n]] for n in kinematics.JOINTS])
    pts = np.array([q0] + [[goal[n] for n in kinematics.JOINTS] for _, goal, _ in moves])
    times = np.concatenate([[0.0], np.cumsum([max(sec, ctx.dt) for _, _, sec in moves])])
    vel = np.zeros_like(pts)  # rest at both ends (and everywhere when not smooth)
    if smooth and len(moves) > 1:
        h = np.diff(times)[:, None]
        d = np.diff(pts, axis=0) / h  # each segment's average speed
        for k in range(1, len(pts) - 1):  # monotone (PCHIP) speeds: no overshoot
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            same = d[k - 1] * d[k] > 0
            vel[k] = np.where(same, (w1 + w2) / np.where(same, w1 / np.where(same, d[k - 1], 1) +
                                                          w2 / np.where(same, d[k], 1), 1), 0.0)
    steps, labels = [], []
    for k, (label, _, _) in enumerate(moves):
        hk = times[k + 1] - times[k]
        n = max(1, int(round(hk / ctx.dt)))
        for i in range(1, n + 1):
            s = i / n  # cubic Hermite between the two poses
            h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
            steps.append(h00 * pts[k] + h10 * hk * vel[k] + h01 * pts[k + 1] + h11 * hk * vel[k + 1])
            labels.append(label)
    ctx.path_steps, ctx.path_labels, ctx.path_i, ctx.path_what = steps, labels, 0, what
    print(f"[{what}] {' -> '.join(dict.fromkeys(labels))} ({len(steps) * ctx.dt:.1f} s"
          f"{', smooth' if smooth and len(moves) > 1 else ''})")


def path(ctx):
    """Y: drive through the poses (cosine-eased joint-space moves), then hold the last
    one. Also plays the R move to HOME."""
    if PATH_KEY in ctx.held:
        ctx.held.discard(PATH_KEY)
        poses = load_poses()
        plan = PATH if PATH else [(n, PATH_SECONDS) for n in poses]
        if not plan:
            print(f"[path] no poses in {POSE_FILE}; press G to save some")
            return False
        missing = [n for n, _ in plan if n not in poses]
        if missing:
            print(f"[path] not in {POSE_FILE}: {missing}")
            return False
        bad = [n for n, _ in plan if not kinematics.in_limits(poses[n]["joints"])]
        if bad:
            print(f"[path] outside the collision-free joint ranges: {bad}; not running")
            return False
        print(f"[path] {POSE_FILE}:")
        start_moves(ctx, [(n, poses[n]["joints"], sec) for n, sec in plan], "path", smooth=PATH_SMOOTH)
    steps = getattr(ctx, "path_steps", None)
    if steps is None:
        return False
    i = ctx.path_i
    for j, n in enumerate(kinematics.JOINTS):
        ctx.targets[ctx.idx[n]] = steps[i][j]
    what = getattr(ctx, "path_what", "path")
    if what == "path" and (i == 0 or ctx.path_labels[i] != ctx.path_labels[i - 1]):
        print(f"[path] -> {ctx.path_labels[i]}")
    ctx.path_i += 1
    if ctx.path_i >= len(steps):
        ctx.path_steps = None
        print(f"[{what}] done")
    return True


def pose_path():
    return os.path.join(os.path.dirname(kinematics.__file__), POSE_FILE)


def load_poses():
    try:
        with open(pose_path()) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_pose(ctx):
    """G: store where the robot actually is (measured angles) as 'live_N' in
    poses.json, with the 5 highest rocks near the pile centre at that moment."""
    q = {n: float(ctx.positions[ctx.idx[n]]) for n in kinematics.JOINTS}
    extra = {}
    if ctx.rocks is not None:
        R = ctx.rocks_module
        pos = ctx.rocks.positions()
        centre = R.SWING_AXIS_XY + R.PILE_DISTANCE * R.BOOM_DIR_XY
        near = np.where(np.linalg.norm(pos[:, :2] - centre, axis=1) < 0.12)[0]
        top = near[np.argsort(-pos[near, 2])][:5]
        extra["rocks_top"] = [[round(float(v), 4) for v in pos[i]] for i in top]
    saved, i = load_poses(), 1
    while f"live_{i}" in saved:  # first unused name, so deleting poses never leads to overwrites
        i += 1
    name = f"live_{i}"
    tip, pitch = kinematics.fk(q)  # same record format as getpose.py
    saved[name] = dict(joints={n: round(v, 5) for n, v in q.items()}, tip=[round(float(v), 5) for v in tip],
                       pitch=round(float(kinematics._near(pitch, 0.0)), 5), source=f"live G at t={ctx.t:.1f} s",
                       **extra)
    with open(pose_path(), "w") as f:
        json.dump(saved, f, indent=2)
    getpose.show(q, f"[pose] {name} -> {POSE_FILE}:")


def pick(ctx):
    """P: plan a pick for the best target rock, then play it back (other keys
    ignored until it ends; the bucket stays up afterwards)."""
    plan = getattr(ctx, "pick_plan", None)
    if PICK_KEY in ctx.held:
        ctx.held.discard(PICK_KEY)
        if ctx.rocks is None:
            print("[pick] no rocks loaded")
            return False
        R = ctx.rocks_module
        centre = R.SWING_AXIS_XY + R.PILE_DISTANCE * R.BOOM_DIR_XY
        q0 = {n: ctx.targets[ctx.idx[n]] for n in kinematics.JOINTS}
        try:
            target, (plan, phases, _) = pick_motion.choose_target(
                ctx.rocks.positions(), centre, q0, ctx.dt, R.terrain_height)
        except ValueError as e:
            print(f"[pick] {e}")
            return False
        ctx.pick_plan, ctx.pick_step, ctx.pick_phases = plan, 0, phases
        print(f"[pick] rock {target}: {len(plan) * ctx.dt:.1f} s motion")
    if plan is None:
        return False
    i = ctx.pick_step
    for n in kinematics.JOINTS:
        ctx.targets[ctx.idx[n]] = plan[i][kinematics.JOINTS.index(n)]
    if i == 0 or ctx.pick_phases[i] != ctx.pick_phases[i - 1]:
        print(f"[pick] {ctx.pick_phases[i]}")
    ctx.pick_step += 1
    if ctx.pick_step >= len(plan):
        ctx.pick_plan = None
        print("[pick] done")
    return True


def sweep(ctx):
    for name, (lo, hi, freq) in JOINT_MOTION.items():
        ctx.targets[ctx.idx[name]] = lo + (hi - lo) * 0.5 * (1 - np.cos(freq * ctx.t))
    return True
