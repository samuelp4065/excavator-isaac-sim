"""Pick up one chosen rock: headless trials of pick_motion.py strategies.

Run with Isaac Sim closed, from the repo root:
    ~/isaacsim/python.sh scripts/pick_rock.py [strategy ...] [--grid] [--grid2] [--targets=N]

Builds the same scene as run_live.py (robot + rocks.py), settles the pile, then
runs each named strategy as one trial per target: robot back home, rocks back
to the settled pile, pick a target rock, play the planned motion, hold, measure.
With no names, the hand-written strategies run; --grid/--grid2 add the parameter
sweeps; --targets=N repeats each on the N best target rocks.

A trial succeeds if the target rock is still in the raised bucket after the hold,
more than LIFT_HEIGHT above the ground. Results go to docs/pick_rock/:
trials.json (one record per trial, appended) and <trial>.npz (time series, see
plot_pick.py). Strategy comments below record why each one was tried.
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import json
import os
import sys
import time

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim, SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import kinematics as K
import pick_motion as M
import rocks as R

USD_PATH = os.path.join(K.REPO, "Excavator_Final_description", "urdf", "Excavator_Final",
                        "Excavator_Final.usd")
OUT_DIR = os.path.join(K.REPO, "docs", "pick_rock")
KP, KD = 1e4, 1e3
LIFT_HEIGHT = 0.10   # m, target rock centre above ground counts as picked
HOLD_SECONDS = 2.0   # rock must stay up this long
PILE_CENTRE = R.SWING_AXIS_XY + R.PILE_DISTANCE * R.BOOM_DIR_XY

# --- Scene ------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
world.get_physics_context().enable_fabric(True)
world.scene.add_default_ground_plane(z_position=R.GROUND_PLANE_Z, static_friction=0.8, dynamic_friction=0.7)
add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/Excavator")
excavator = world.scene.add(SingleArticulation("/World/Excavator", name="excavator"))
rockset = R.RockSet(world.stage)
world.reset()
excavator.get_articulation_controller().set_gains(kps=np.full(4, KP), kds=np.full(4, KD))
names = excavator.dof_names
bucket = RigidPrim("/World/Excavator/Bucket_1", name="bucket_link")
bucket.initialize()
rockset.initialize()
DT = world.get_physics_dt()
targets = np.zeros(4)


def hold():
    excavator.apply_action(ArticulationAction(joint_positions=targets))


rockset.settle(world, before_step=hold)

# Cutting-edge tracking, as in check_kinematics.py: a fixed point on the bucket link.
excavator.set_joint_positions(targets)
hold()
world.step(render=False)
TIP_REF = K.fk(dict(zip(names, excavator.get_joint_positions())))[0]
_p, _q = bucket.get_world_poses()
P0, R0 = _p[0], Rotation.from_quat([_q[0][1], _q[0][2], _q[0][3], _q[0][0]])


def sim_tip():
    p, q = bucket.get_world_poses()
    Rb = Rotation.from_quat([q[0][1], q[0][2], q[0][3], q[0][0]])
    return (Rb * R0.inv()).apply(TIP_REF - P0) + p[0]


# --- Strategies -----------------------------------------------------------------
# Each takes (rock xyz, start joints) and returns (targets (N, 4) in K.JOINTS
# order, phase per step); the harness plays them back and measures.

def replay_verified(rock, q0):
    """Baseline: the joint-space dig cycle verified on an earlier pile (not aimed at a rock)."""
    p = M._Plan(q0, DT)
    for phase, q, s in [("approach", (0, -0.73, -1.55, -1.68), 3.0), ("drag", (0, 0.01, 1.19, -3.67), 4.0),
                        ("curl", (0, 0.01, 1.19, -1.67), 2.0), ("lift", (0, 0.41, 1.19, -1.27), 2.0)]:
        p.phase = phase
        p.joints(K.as_dict(q), s)
    return np.array(p.targets), p.phases


def grab_from_above(rock, q0):
    """Naive: lower the open bucket (opening up) onto the rock, then tilt it up."""
    p = M._Plan(q0, DT)
    p.phase = "approach"
    p.joints(K.ik(rock + [0, 0, 0.15], M.PITCH_CARRY), 3.0)
    p.phase = "lower"
    p.tip_line(rock, M.PITCH_CARRY, 2.0)
    p.phase = "curl"
    p.tip_line(rock + [0, 0, 0.03], M.PITCH_CARRY - 0.6, 1.5)
    p.phase = "lift"
    p.tip_line(rock + [0, 0, 0.2], M.PITCH_CARRY - 0.6, 2.0)
    return np.array(p.targets), p.phases


# Parameters of the first drag attempts (pick_motion.DEFAULTS are the final ones).
EARLY = dict(behind=0.06, past=0.05, depth=0.008, drag_s=3.0, curl_mode="joint", pitch=M.PITCH_DIG,
             clearance=0.005, curl_advance=0.0, curl_s=2.0, follow_terrain=False, carry=M.PITCH_CARRY)


def make_drag(**params):
    p = {**EARLY, **params}

    def strategy(rock, q0):
        t, ph, _ = M.plan_pick(rock, q0, DT, terrain=R.terrain_height, **p)
        return t, ph
    strategy.params = p
    return strategy


STRATEGIES = {
    # Round 1, crater pile.
    "replay_verified": replay_verified,
    "grab_from_above": grab_from_above,
    "drag_v1": make_drag(),
    # v1 shoved the rock along: the edge was only 8 mm below its centre.
    "drag_v2_deep": make_drag(depth=0.025),
    "drag_v3_flat": make_drag(depth=0.025, pitch=2.0),
    "drag_v4_edgecurl": make_drag(depth=0.025, curl_mode="edge"),
    "drag_v5_flat_edgecurl": make_drag(depth=0.025, pitch=2.0, curl_mode="edge"),
    # v4 lifted 4 rocks but the curl drove the bucket heel into the ground and jammed.
    # v6/v7 (heel kept off flat ground only) still jammed: the bucket's back went into
    # the crater wall. Now the clearance check covers every bucket vertex vs the crater.
    "drag_v6_clearcurl": make_drag(depth=0.025, curl_mode="clear"),
    "drag_v7_clearcurl_advance": make_drag(depth=0.025, curl_mode="clear", curl_advance=0.04),
}
# v7 no longer jams, but the rock is bulldozed ahead of the lip and drops off the
# front of the heap before the curl. Grid over a shorter drag + quicker curl that
# pushes the lip forward under the rock: names like grid_b0.05_p0.01_a0.06_d0.025_c1.0.
GRID = [dict(behind=b, past=p, curl_advance=a, depth=d, curl_s=c)
        for b in (0.04, 0.06) for p in (-0.01, 0.01, 0.03) for a in (0.04, 0.08)
        for d in (0.025, 0.035) for c in (1.0, 2.0)]
for g in GRID:
    STRATEGIES["grid_b{behind}_p{past}_a{curl_advance}_d{depth}_c{curl_s}".format(**g)] = \
        make_drag(curl_mode="clear", **g)
# Grid 1: 3/48 succeeded, all with the 1 s curl, 4 cm advance and the edge at ground
# level; neighbours failed and the same rock sat 1-3 mm differently between resets,
# so grid 2 would be judged over several target rocks (not run; replaced by the pit).
GRID2 = [dict(behind=b, past=p, curl_advance=a, curl_s=c)
         for b in (0.04, 0.06) for p in (0.0, 0.01, 0.02) for a in (0.03, 0.04, 0.05) for c in (0.6, 1.0)]
for g in GRID2:
    STRATEGIES["grid2_b{behind}_p{past}_a{curl_advance}_c{curl_s}".format(**g)] = \
        make_drag(curl_mode="clear", depth=0.05, **g)

# Round 2, sunken pit (rocks.PILE_STYLE = "pit"): rocks sit below ground level,
# held by the pit walls, so the lip can pass under them instead of bulldozing them.
PIT = dict(curl_mode="clear", follow_terrain=True, curl_advance=0.04, curl_s=1.0)
STRATEGIES.update({
    "pit_v1": make_drag(behind=0.06, past=0.03, depth=0.02, **PIT),
    "pit_v2_deeper": make_drag(behind=0.06, past=0.03, depth=0.03, **PIT),
    "pit_v3_long": make_drag(behind=0.08, past=0.05, depth=0.03, **PIT),
    "pit_v4_long_fastcurl": make_drag(behind=0.08, past=0.05, depth=0.03, **{**PIT, "curl_s": 0.6}),
    # v1-v4 scoop 12-27 rocks, but the target rides on top of the heaped load and
    # spills over the lip during the lift (v4 t1: up to 28 cm, then out). Also the
    # far entry points were only reachable by swinging round (now refused).
    # Lower pitch while lifting = lip held higher.
    "pit_v5_tiltback": make_drag(behind=0.08, past=0.05, depth=0.03, **{**PIT, "curl_s": 0.6}, carry=0.1),
    "pit_v6_tiltback_more": make_drag(behind=0.08, past=0.05, depth=0.03, **{**PIT, "curl_s": 0.6}, carry=-0.2),
    "pit_v7_short_tiltback": make_drag(behind=0.06, past=0.03, depth=0.03, **PIT, carry=0.1),
    # Moving the pit to 0.39 m (to fix reach) made it worse: the bucket couldn't push
    # down into the packed, walled-in rocks. Back at 0.45 m; v6 = pick_motion.DEFAULTS.
    "final": make_drag(**M.DEFAULTS),
})


# --- Run ------------------------------------------------------------------------

def run_trial(name, rank=0):
    global targets
    targets = np.zeros(4)
    excavator.set_joint_positions(targets)
    excavator.set_joint_velocities(np.zeros(4))
    rockset.reset()
    for _ in range(30):
        hold()
        world.step(render=False)
    pos = rockset.positions()
    q0 = dict(zip(names, targets))
    # Same target for every strategy: the final motion must be plannable for it.
    target, _ = M.choose_target(pos, PILE_CENTRE, q0, DT, R.terrain_height, rank)
    rock = pos[target].copy()
    trial_name = name + (f"__t{rank}" if rank else "")
    error = None
    wall = time.time()
    try:
        plan, phases = STRATEGIES[name](rock, q0)
    except ValueError as e:  # IK out of reach
        plan, phases, error = np.zeros((0, 4)), [], str(e)
    plan = list(plan) + [plan[-1]] * int(HOLD_SECONDS / DT) if len(plan) else []
    phases = list(phases) + ["hold"] * (len(plan) - len(phases))
    L = {k: [] for k in ("t", "q_cmd", "q", "tip", "target_pos", "n_up", "phase", "clear_cmd")}
    for i, (q_cmd, phase) in enumerate(zip(plan, phases)):
        targets = K.as_array(K.as_dict(q_cmd), names)
        hold()
        world.step(render=False)
        pos = rockset.positions()
        L["t"].append((i + 1) * DT)
        L["q_cmd"].append(targets.copy())
        L["q"].append(excavator.get_joint_positions().copy())
        L["tip"].append(sim_tip())
        L["target_pos"].append(pos[target].copy())
        L["n_up"].append(int((pos[:, 2] > LIFT_HEIGHT).sum()))
        L["phase"].append(phase)
        L["clear_cmd"].append(M.bucket_clearance(dict(zip(names, targets)), R.terrain_height))
    L = {k: np.array(v) for k, v in L.items()}
    final = rockset.positions()
    ran = len(L["t"]) > 0
    tz = L["target_pos"][:, 2] if ran else np.array([rock[2]])
    lifted_idx = np.where(final[:, 2] > LIFT_HEIGHT)[0]
    tip_err = (np.linalg.norm(L["tip"] - np.array([K.fk(dict(zip(names, q)))[0] for q in L["q_cmd"]]), axis=1)
               if ran else np.array([np.nan]))
    rec = dict(
        strategy=name,
        params=getattr(STRATEGIES[name], "params", None),
        when=time.strftime("%Y-%m-%d %H:%M:%S"),
        target=target,
        target_rank=rank,
        scene=R.PILE_STYLE,
        target_start=rock.round(4).tolist(),
        target_final=final[target].round(4).tolist(),
        target_max_z=float(tz.max()),
        success=bool(final[target, 2] > LIFT_HEIGHT and error is None),
        rocks_up_at_end=int(len(lifted_idx)),
        any_rock_success=bool(len(lifted_idx) > 0 and error is None),
        rocks_up_max=int(L["n_up"].max()) if ran else 0,
        target_moved_xy=float(np.linalg.norm(final[target, :2] - rock[:2])),
        tip_tracking_mean_mm=float(np.nanmean(tip_err) * 1000),
        tip_tracking_max_mm=float(np.nanmax(tip_err) * 1000),
        joint_err_max=float(np.abs(L["q"] - L["q_cmd"]).max()) if ran else None,
        min_clearance_by_phase={p: round(float(L["clear_cmd"][L["phase"] == p].min()), 4)
                                for p in dict.fromkeys(L["phase"].tolist())} if ran else {},
        sim_seconds=round(len(L["t"]) * DT, 2),
        wall_seconds=round(time.time() - wall, 1),
        error=error,
    )
    if ran:
        np.savez_compressed(os.path.join(OUT_DIR, f"{trial_name}.npz"), **L, rock_start=rock, rocks_final=final,
                            scene=R.PILE_STYLE)
    return rec


os.makedirs(OUT_DIR, exist_ok=True)
wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
for flag, prefix in (("--grid", "grid_"), ("--grid2", "grid2_")):
    if flag in sys.argv:
        wanted += [n for n in STRATEGIES if n.startswith(prefix)]
wanted = wanted or [n for n in STRATEGIES if not n.startswith("grid")]
n_targets = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--targets=")), 1))
results = []
for name, rank in [(n, k) for n in wanted for k in range(n_targets)]:
    rec = run_trial(name, rank)
    results.append(rec)
    with open(os.path.join(OUT_DIR, "trials.json"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[pick] {name} t{rank}: {'SUCCESS' if rec['success'] else 'fail'} "
          f"target max z {rec['target_max_z'] * 100:.1f} cm, final {rec['target_final']}, "
          f"rocks up {rec['rocks_up_at_end']} (max {rec['rocks_up_max']}), err={rec['error']}", flush=True)

with open(os.path.join(OUT_DIR, "last_run.txt"), "w") as f:
    f.write("\n".join(json.dumps(r) for r in results) + "\n")
simulation_app.close()
