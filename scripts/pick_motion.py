"""Plan a bucket motion that scoops up one chosen rock (pure numpy, no Isaac Sim).

The motion is open loop: every joint target comes from kinematics.py, none from
sim feedback, so it is planned up front as one array of per-step targets and
played back by pick_rock.py (headless trials) or live_controller.py (P key).

Edge-first drag aimed at the rock (pitch in kinematics.py's convention,
arm - pole - bucket; 2.5 = cutting edge pointing down and back at the machine,
0.5 = bucket opening straight up):
    approach  joint move to above the entry point, `behind` m beyond the rock
    plunge    edge straight down to `depth` m below the rock's centre (or as deep
              as keeps the whole bucket out of the dirt)
    drag      edge pulled toward the machine to `past` m short of the rock
    curl      rotate the bucket up about the edge, edge moving `curl_advance` m
              further in, raised only as far as keeps the whole bucket out of the dirt
    lift      boom up `lift_arm` rad (~0.45 = 26 deg) while tilting to `carry`
              (below 0.5 = lip higher), or as far as the bucket turns before the stick
All joint targets stay inside kinematics.JOINT_LIMITS, so no link passes
through another (the sim itself doesn't stop neighbouring links overlapping).
Defaults are the best of the trials in docs/pick_rock/ (sunken pit scene).

    python3 scripts/pick_motion.py    # plans a pick for a sample rock, prints phases
"""
import numpy as np

import kinematics as K

PITCH_DIG = 2.5
PITCH_CARRY = 0.5
DEFAULTS = dict(behind=0.08, past=0.05, depth=0.03, drag_s=3.0, curl_mode="clear", pitch=PITCH_DIG,
                clearance=0.005, curl_advance=0.04, curl_s=0.6, follow_terrain=True, carry=-0.2,
                approach_s=3.0, plunge_s=1.5, lift_s=2.5, lift_arm=0.45)

_BUCKET_PTS = K._read_stl(K.BUCKET_STL)                          # zero-pose mesh vertices (world = swing 0)
_BUCKET_UZ = np.array([K._to_plane(p)[0] for p in _BUCKET_PTS])  # ... in the boom plane
_BUCKET_W = (_BUCKET_PTS - K.SWING_ORIGIN) @ K.AXIS              # ... and their sideways offsets


def flat_ground(xy):
    return np.zeros(len(np.atleast_2d(xy)))


def bucket_clearance(q, terrain=flat_ground):
    """Smallest gap (m) between the bucket mesh and the dirt for joint angles q;
    negative means the bucket would be inside it. terrain(xy (n, 2)) -> heights."""
    pts, pitch = K.fk_plane(q["arm_joint"], q["pole_joint"], q["bucket_joint"])
    uz = (K._rot(pitch) @ (_BUCKET_UZ - K.BUCKET_PIN).T).T + pts["bucket_pin"]
    sw = q["swing_joint"]
    horiz = uz[:, :1] * K.U_HAT + _BUCKET_W[:, None] * K.AXIS
    c, s = np.cos(sw), np.sin(sw)
    xy = K.SWING_ORIGIN[:2] + np.stack([c * horiz[:, 0] - s * horiz[:, 1], s * horiz[:, 0] + c * horiz[:, 1]], 1)
    return float((uz[:, 1] - terrain(xy)).min())


def ik_fwd(tip, pitch, q_ref=None):
    """kinematics.ik, but only the boom-forward branch: near the edge of reach the
    other solution swings round 180 degrees and reaches back over the cab."""
    q = K.ik(tip, pitch, q_ref=q_ref)
    if abs(K._near(q["swing_joint"], 0.0)) > np.pi / 2:
        q = K.ik(tip, pitch)  # the forward branch may exist but be farther from q_ref
        if abs(K._near(q["swing_joint"], 0.0)) > np.pi / 2:
            raise ValueError(f"{np.round(tip, 3)} only reachable by swinging round")
        q = K.wrap(q, q_ref)
    return q


def ik_clear(xy, z_min, pitch, clearance, q_ref, terrain=flat_ground):
    """IK for the edge at xy and this pitch, at height z_min or the lowest height
    above it that keeps the whole bucket `clearance` above the dirt."""
    q = None
    for dz in np.arange(0.0, 0.30, 0.002):
        try:
            q = ik_fwd(np.array([*xy, z_min + dz]), pitch, q_ref=q_ref)
        except ValueError:  # out of reach this low; higher may work
            continue
        if bucket_clearance(q, terrain) >= clearance:
            return q
    if q is None:
        raise ValueError(f"edge at {np.round(xy, 3)} pitch {pitch:.2f} out of reach")
    return q


def radial(xy):
    """Unit vector in the ground plane pointing away from the swing axis."""
    d = np.asarray(xy) - K.SWING_ORIGIN[:2]
    return d / np.linalg.norm(d)


def choose_target(pos, centre, q_start, dt, terrain=flat_ground, rank=0, radius=0.06, **params):
    """Pick a top-of-heap rock within `radius` of the pile centre (easiest to get
    under) whose whole motion plans without running out of reach; rank 1, 2, ...
    give the next candidates. Returns (rock index, plan_pick's result)."""
    d = np.linalg.norm(pos[:, :2] - centre, axis=1)
    score = np.where(d < radius, pos[:, 2] - 0.2 * d, -np.inf)
    found = 0
    for i in np.argsort(-score):
        if not np.isfinite(score[i]):
            break
        try:
            plan = plan_pick(pos[i], q_start, dt, terrain, **params)
        except ValueError:
            continue
        if found == rank:
            return int(i), plan
        found += 1
    raise ValueError("no reachable rock near the pile centre")


class _Plan:
    """Accumulates per-step joint targets (dicts) and phase labels."""

    def __init__(self, q_start, dt):
        self.q = K.as_dict(q_start)
        self.dt = dt
        self.targets, self.phases = [], []
        self.phase = ""

    def _push(self, q):
        self.q = q
        self.targets.append(K.as_array(q))
        self.phases.append(self.phase)

    def _steps(self, seconds):
        n = max(1, int(seconds / self.dt))
        return [(0.5 * (1 - np.cos(np.pi * i / n))) for i in range(1, n + 1)]

    def joints(self, q_goal, seconds):
        """Cosine-eased straight line in joint space."""
        q0 = K.as_array(self.q)
        q1 = K.as_array(K.wrap(q_goal, self.q))  # the in-range way round, never through a link
        for s in self._steps(seconds):
            self._push(K.as_dict(q0 + s * (q1 - q0)))

    def tip_line(self, p1, pitch1, seconds):
        """Cutting edge along a straight line (and pitch blended), IK every step."""
        p0, pitch0 = K.fk(self.q)
        pitch1 = K._near(pitch1, pitch0)  # pitch is an angle: blend the short way
        for s in self._steps(seconds):
            self._push(ik_fwd(p0 + s * (np.asarray(p1) - p0), pitch0 + s * (pitch1 - pitch0), q_ref=self.q))


def plan_pick(rock, q_start, dt, terrain=flat_ground, **params):
    """Joint targets to scoop up the rock at xyz `rock`, starting from q_start.

    Returns (targets (N, 4) in kinematics.JOINTS order, phase label per step,
    params used). terrain(xy (n, 2)) -> dirt heights (e.g. rocks.terrain_height).
    Raises ValueError if the rock can't be reached."""
    p = {**DEFAULTS, **params}
    rock = np.asarray(rock, float)
    u = radial(rock[:2])
    z_cut = rock[2] - p["depth"]

    def edge_z(xy):  # planned edge height: depth below the rock, never in the dirt
        return max(z_cut, terrain(xy)[0] + 0.008)

    # Enter `behind` beyond the rock, or as far as reach allows (far side of the pit).
    for b in np.arange(p["behind"], 0.015, -0.01):
        xy0 = rock[:2] + b * u
        try:
            ik_fwd(np.array([*xy0, edge_z(xy0)]), p["pitch"])
            break
        except ValueError:
            continue
    xy1 = rock[:2] - p["past"] * u
    p0 = np.array([*xy0, edge_z(xy0)])
    p1 = np.array([*xy1, edge_z(xy1)])

    plan = _Plan(q_start, dt)
    plan.phase = "approach"
    for h in np.arange(0.12, 0.0, -0.01):  # as high above the entry point as reach allows
        try:
            q_above = ik_fwd(p0 + [0, 0, h], p["pitch"])
            break
        except ValueError:
            continue
    else:
        raise ValueError("entry point out of reach")
    plan.joints(q_above, p["approach_s"])
    plan.phase = "plunge"
    if p["follow_terrain"]:  # straight down, but never deeper than keeps the whole bucket clear
        top = K.fk(plan.q)[0][2]
        for s in plan._steps(p["plunge_s"]):
            plan._push(ik_clear(xy0, top + s * (p0[2] - top), p["pitch"], p["clearance"], plan.q, terrain))
    else:
        plan.tip_line(p0, p["pitch"], p["plunge_s"])
    plan.phase = "drag"
    if p["follow_terrain"]:  # edge follows the dirt profile (pit walls), whole bucket kept clear
        for s in plan._steps(p["drag_s"]):
            xy = xy0 + s * (xy1 - xy0)
            plan._push(ik_clear(xy, edge_z(xy), p["pitch"], p["clearance"], plan.q, terrain))
        p1 = np.array([*xy1, K.fk(plan.q)[0][2]])
    else:
        plan.tip_line(p1, p["pitch"], p["drag_s"])
    plan.phase = "curl"
    if p["curl_mode"] == "joint":  # bucket joint only
        q = dict(plan.q)
        q["bucket_joint"] += p["pitch"] - PITCH_CARRY
        plan.joints(q, p["curl_s"])
    elif p["curl_mode"] == "edge":  # rotate about the edge while raising it a little
        plan.tip_line(p1 + [0, 0, 0.04], PITCH_CARRY, p["curl_s"])
    else:  # "clear": rotate about the edge, raising it only as far as keeps the bucket out of the dirt
        for s in plan._steps(p["curl_s"]):
            plan._push(ik_clear(p1[:2] - s * p["curl_advance"] * u, p1[2],
                                p["pitch"] + s * (PITCH_CARRY - p["pitch"]), p["clearance"], plan.q, terrain))
    plan.phase = "lift"  # raise the boom; tilt back to `carry`, or as far as the bucket's range allows
    q = dict(plan.q)
    q["arm_joint"] = min(q["arm_joint"] + p["lift_arm"], K.JOINT_LIMITS["arm_joint"][1])
    lo, hi = K.JOINT_LIMITS["bucket_joint"]
    bucket = lo + (q["arm_joint"] - q["pole_joint"] - p["carry"] - lo) % (2 * np.pi)
    q["bucket_joint"] = float(np.clip(bucket, lo, hi - np.radians(3)))
    plan.joints(q, p["lift_s"])
    return np.array(plan.targets), plan.phases, p


if __name__ == "__main__":
    rock = np.array([0.068, -0.40, -0.004])
    targets, phases, _ = plan_pick(rock, np.zeros(4), 1 / 60)
    print(f"{len(targets)} steps ({len(targets) / 60:.1f} s at 60 Hz) to pick a rock at {rock}")
    for ph in dict.fromkeys(phases):
        i = len(phases) - 1 - phases[::-1].index(ph)
        tip, pitch = K.fk(targets[i])
        print(f"{ph:9s} ends at step {i:4d}: edge {np.round(tip, 3)}, pitch {pitch:+.2f}")
