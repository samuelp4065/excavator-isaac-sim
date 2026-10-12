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
"""
import carb.input
import numpy as np

MODE = "keyboard"  # "keyboard" or "sweep"

# Drive gains, re-applied on every reload.
KP = 1e4
KD = 1e3

K = carb.input.KeyboardInput
# All joints are 'continuous' in the URDF, so these clamps (radians) are the only
# bound on motion. Kept wide on purpose so you can explore; tighten once you know.
JOINT_KEYS = {
    # name:          (key +,   key -,   lo,     hi)
    "swing_joint":  (K.LEFT,  K.RIGHT, -np.pi, np.pi),
    "arm_joint":    (K.Q,     K.A,     -np.pi, np.pi),
    "pole_joint":   (K.W,     K.S,     -np.pi, np.pi),
    "bucket_joint": (K.E,     K.D,     -np.pi, np.pi),
}
RESET_KEY = K.R
SPEED = 0.5  # rad/s while a key is held (~29 deg/s)

# Used when MODE == "sweep": each joint goes lo -> hi -> lo at freq (rad/s).
JOINT_MOTION = {
    # name:          (lo,    hi,   freq)
    "swing_joint":  (0.0,   np.pi, 0.3),
    "arm_joint":    (0.0,   0.8,   0.5),
    "pole_joint":   (-0.4,  0.4,   0.7),
    "bucket_joint": (-0.5,  0.5,   0.9),
}

PRINT_EVERY = 6  # steps between angle printouts while moving


def step(ctx):
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


def keyboard(ctx):
    moved = False
    if RESET_KEY in ctx.held:
        ctx.targets[:] = 0.0
        moved = True
    for name, (key_up, key_down, lo, hi) in JOINT_KEYS.items():
        direction = (key_up in ctx.held) - (key_down in ctx.held)
        if direction:
            i = ctx.idx[name]
            ctx.targets[i] = np.clip(ctx.targets[i] + direction * SPEED * ctx.dt, lo, hi)
            moved = True
    return moved


def sweep(ctx):
    for name, (lo, hi, freq) in JOINT_MOTION.items():
        ctx.targets[ctx.idx[name]] = lo + (hi - lo) * 0.5 * (1 - np.cos(freq * ctx.t))
    return True
