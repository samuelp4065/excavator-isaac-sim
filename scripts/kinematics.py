"""Forward and inverse kinematics of the excavator's bucket tip.

Plain numpy (no Isaac Sim), so it can be studied and tested with system Python:
    python3 scripts/kinematics.py          # runs the math self-test
and checked against the simulator with:
    ~/isaacsim/python.sh scripts/check_kinematics.py

How the robot is modelled
-------------------------
Swing turns everything about the vertical axis. Boom (arm_joint), stick
(pole_joint) and bucket (bucket_joint) all rotate about parallel horizontal axes,
so they move in one vertical plane: a 3-link planar arm on a turntable.

Inside that plane we use 2D coordinates (u, z):
    u = horizontal distance outward along the boom, z = height.
A positive angle turns counter-clockwise when looking with u to the right and z
up, i.e. it raises the part. The boom's URDF axis points the opposite way to the
stick's and bucket's, so in plane angles:
    theta1 = +arm,  theta2 = -pole,  theta3 = -bucket
and the bucket's pitch (its rotation from the zero pose) is
    pitch = theta1 + theta2 + theta3 = arm - pole - bucket.

All geometry is read from Excavator_Final.urdf (pin positions, axes) and the
bucket mesh (cutting edge), so it follows edits to those files. At the zero pose
the meshes sit in world coordinates (the URDF has no joint rotations), which is
what makes reading positions straight from them valid.
"""
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
URDF_PATH = os.path.join(REPO, "Excavator_Final_description", "urdf", "Excavator_Final.urdf")
BUCKET_STL = os.path.join(REPO, "Excavator_Final_description", "meshes", "Bucket_1.stl")
JOINTS = ("swing_joint", "arm_joint", "pole_joint", "bucket_joint")

# Collision-free joint ranges (rad), measured by rotating each link mesh against
# its parent (ignoring 3 cm around the pin, where the hinge bosses overlap). The
# URDF joints are 'continuous' and the sim doesn't collide neighbouring links, so
# nothing else stops e.g. the bucket swinging through the stick. The bucket hits
# the stick between -80 and -10 deg, so its range runs the long way round, from
# -8 up to +278 deg: a pose at -160 deg must be reached as +200 deg. Swing is free.
JOINT_LIMITS = {
    "arm_joint": (np.radians(-50), np.radians(110)),    # boom vs cab
    "pole_joint": (np.radians(-5), np.radians(130)),    # stick vs boom
    "bucket_joint": (np.radians(-8), np.radians(278)),  # bucket vs stick
}


# --- Geometry, read once at import --------------------------------------------

def _read_urdf_joints(path):
    joints = {}
    for j in ET.parse(path).getroot().iter("joint"):
        joints[j.get("name")] = dict(
            xyz=np.array([float(v) for v in j.find("origin").get("xyz").split()]),
            axis=np.array([float(v) for v in j.find("axis").get("xyz").split()]),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
        )
    return joints


def _world_pins(joints):
    """World position of every joint at the zero pose (origins chain from base_link)."""
    link_pos = {"base_link": np.zeros(3)}
    pins = {}
    while len(pins) < len(joints):
        for name, j in joints.items():
            if name not in pins and j["parent"] in link_pos:
                pins[name] = link_pos[j["parent"]] + j["xyz"]
                link_pos[j["child"]] = pins[name]
    return pins


def _read_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)
    return np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0) * 0.001  # mm -> m


_J = _read_urdf_joints(URDF_PATH)
_PINS = _world_pins(_J)

AXIS = _J["arm_joint"]["axis"] / np.linalg.norm(_J["arm_joint"]["axis"])   # boom axis (horizontal)
U_HAT = np.cross([0.0, 0.0, 1.0], AXIS)                                    # outward along the boom
SWING_ORIGIN = _PINS["swing_joint"]                                        # on the vertical swing axis


def _to_plane(p):
    """World point at swing=0 -> (u, z) in the boom plane, plus its offset w along the axis."""
    d = p - SWING_ORIGIN
    return np.array([d @ U_HAT, p[2]]), d @ AXIS


ARM_PIN, _ = _to_plane(_PINS["arm_joint"])        # boom pivot
POLE_PIN, _ = _to_plane(_PINS["pole_joint"])      # stick pivot
BUCKET_PIN, _ = _to_plane(_PINS["bucket_joint"])  # bucket pivot


def _bucket_tip():
    """Middle of the bucket's cutting edge: the mesh points farthest from the bucket pin."""
    pts = _read_stl(BUCKET_STL)
    plane = np.array([_to_plane(p)[0] for p in pts])
    w = (pts - SWING_ORIGIN) @ AXIS
    dist = np.linalg.norm(plane - BUCKET_PIN, axis=1)
    edge = dist > dist.max() - 0.002
    return plane[edge].mean(axis=0), w[edge].mean()


TIP0, TIP_W = _bucket_tip()   # tip in the plane at the zero pose; TIP_W = its sideways offset

L1 = np.linalg.norm(POLE_PIN - ARM_PIN)       # boom pivot -> stick pivot
L2 = np.linalg.norm(BUCKET_PIN - POLE_PIN)    # stick pivot -> bucket pivot
L3 = np.linalg.norm(TIP0 - BUCKET_PIN)        # bucket pivot -> cutting edge
_B1 = np.arctan2(*(POLE_PIN - ARM_PIN)[::-1])     # zero-pose direction of the boom
_B2 = np.arctan2(*(BUCKET_PIN - POLE_PIN)[::-1])  # zero-pose direction of the stick
_ELBOW = np.sign(np.sin(_B2 - _B1)) or 1.0        # which IK branch the real machine uses


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# --- Forward kinematics -------------------------------------------------------

def fk_plane(arm, pole, bucket):
    """Points of the arm in the boom plane (u, z): boom pivot, stick pivot, bucket
    pivot, cutting edge. Returns a dict, plus the bucket pitch."""
    t1, t2, t3 = arm, -pole, -bucket
    stick = ARM_PIN + _rot(t1) @ (POLE_PIN - ARM_PIN)
    bpin = stick + _rot(t1 + t2) @ (BUCKET_PIN - POLE_PIN)
    tip = bpin + _rot(t1 + t2 + t3) @ (TIP0 - BUCKET_PIN)
    return dict(arm_pin=ARM_PIN.copy(), pole_pin=stick, bucket_pin=bpin, tip=tip), t1 + t2 + t3


def plane_to_world(uz, swing, w=TIP_W):
    """(u, z) in the boom plane at sideways offset w -> world xyz, after swinging."""
    horiz = uz[0] * U_HAT + w * AXIS
    c, s = np.cos(swing), np.sin(swing)
    xy = SWING_ORIGIN[:2] + np.array([c * horiz[0] - s * horiz[1], s * horiz[0] + c * horiz[1]])
    return np.array([xy[0], xy[1], uz[1]])


def fk(q):
    """Joint angles -> (bucket tip world xyz, bucket pitch).
    q: dict {joint name: angle} or a sequence (swing, arm, pole, bucket)."""
    q = as_dict(q)
    pts, pitch = fk_plane(q["arm_joint"], q["pole_joint"], q["bucket_joint"])
    return plane_to_world(pts["tip"], q["swing_joint"]), pitch


# --- Inverse kinematics -------------------------------------------------------

def _near(angle, ref):
    """angle + 2*pi*k closest to ref."""
    return ref + (angle - ref + np.pi) % (2 * np.pi) - np.pi


def wrap(q, ref=None):
    """Pick each joint's equivalent angle (+ 2*pi*k): inside JOINT_LIMITS for the
    limited joints, nearest ref (default 0) for the rest. Returns None if a
    limited joint has no equivalent inside its range."""
    q, ref = as_dict(q), as_dict(ref) if ref is not None else dict.fromkeys(JOINTS, 0.0)
    out = {}
    for n in JOINTS:
        if n in JOINT_LIMITS:
            lo, hi = JOINT_LIMITS[n]
            a = lo + (q[n] - lo) % (2 * np.pi)
            if a > hi:
                return None
            out[n] = a
        else:
            out[n] = _near(q[n], ref[n])
    return out


def in_limits(q):
    q = as_dict(q)
    return all(lo <= q[n] <= hi for n, (lo, hi) in JOINT_LIMITS.items())


def ik(tip, pitch, q_ref=None):
    """Bucket tip world xyz + bucket pitch -> joint angles dict.

    Angles are put inside JOINT_LIMITS; swing is wrapped to the nearest
    equivalent of q_ref (optional, e.g. the current pose) so it never spins a
    full turn. Raises ValueError if the target is out of reach or needs a joint
    outside its range."""
    tip = np.asarray(tip, float)
    ref = as_dict(q_ref) if q_ref is not None else dict.fromkeys(JOINTS, 0.0)

    # Swing: turn so the boom plane passes through the target. The tip sits TIP_W
    # to the side of the plane, so the in-plane reach u satisfies u^2 + w^2 = d^2.
    # u can be positive (boom reaching out) or negative (target behind the swing
    # axis, reached by swinging the other way); try both, keep the closest to ref.
    d_vec = tip[:2] - SWING_ORIGIN[:2]
    d = np.linalg.norm(d_vec)
    if d < abs(TIP_W):
        raise ValueError("target is too close to the swing axis")
    best, err = None, None
    for sign in (1.0, -1.0):
        u = sign * np.sqrt(d ** 2 - TIP_W ** 2)
        zero_dir = u * U_HAT[:2] + TIP_W * AXIS[:2]
        swing = np.arctan2(d_vec[1], d_vec[0]) - np.arctan2(zero_dir[1], zero_dir[0])
        try:
            q = _ik_plane(np.array([u, tip[2]]), pitch)
        except ValueError as e:
            err = e
            continue
        q["swing_joint"] = swing
        q = wrap(q, ref)
        if q is None:
            err = ValueError("target needs a joint outside its collision-free range")
            continue
        if best is None or _dist(q, ref) < _dist(best, ref):
            best = q
    if best is None:
        raise err
    return best


def _dist(q, ref):
    return sum((q[n] - ref[n]) ** 2 for n in JOINTS)


def _ik_plane(target, pitch):
    """Tip (u, z) in the boom plane + pitch -> boom, stick, bucket angles."""
    # Bucket pivot: step back from the tip along the bucket.
    bpin = target - _rot(pitch) @ (TIP0 - BUCKET_PIN)

    # Boom + stick: a 2-link arm from the boom pivot to the bucket pivot (law of cosines).
    D = bpin - ARM_PIN
    dist = np.linalg.norm(D)
    if not abs(L1 - L2) <= dist <= L1 + L2:
        raise ValueError(f"bucket pivot {dist:.3f} m from the boom pivot; reachable "
                         f"{abs(L1 - L2):.3f}-{L1 + L2:.3f} m")
    cos_g = np.clip((dist ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
    g = _ELBOW * np.arccos(cos_g)                              # stick angle relative to boom
    a1 = np.arctan2(D[1], D[0]) - np.arctan2(L2 * np.sin(g), L1 + L2 * np.cos(g))
    a2 = a1 + g                                                # absolute stick direction

    t1 = a1 - _B1
    t2 = a2 - _B2 - t1
    t3 = pitch - t1 - t2
    return {"arm_joint": t1, "pole_joint": -t2, "bucket_joint": -t3}


# --- Helpers ----------------------------------------------------------------------

def as_dict(q):
    if isinstance(q, dict):
        return q
    return dict(zip(JOINTS, (float(v) for v in q)))


def as_array(q, dof_names=JOINTS):
    """Joint dict -> array in the articulation's DOF order (e.g. excavator.dof_names)."""
    return np.array([q[n] for n in dof_names])


def describe():
    return (f"links: boom {L1:.3f} m, stick {L2:.3f} m, bucket {L3:.3f} m; "
            f"tip at zero pose u={TIP0[0]:.3f} z={TIP0[1]:.3f} (side offset {TIP_W * 1000:.0f} mm); "
            f"elbow branch {_ELBOW:+.0f}")


if __name__ == "__main__":
    # Math self-test: ik(fk(q)) must give back the same tip and pitch.
    print(describe())
    rng = np.random.default_rng(0)
    worst, tested = 0.0, 0
    for _ in range(2000):
        q = dict(zip(JOINTS, rng.uniform(-1.2, 1.2, 4)))
        q.update({n: rng.uniform(lo, hi) for n, (lo, hi) in JOINT_LIMITS.items()})
        tip, pitch = fk(q)
        q2 = ik(tip, pitch, q_ref=q)
        tip2, pitch2 = fk(q2)
        worst = max(worst, np.linalg.norm(tip2 - tip), abs(_near(pitch2, pitch) - pitch))
        tested += 1
    print(f"fk -> ik -> fk on {tested} random poses: worst error {worst:.2e}")
    raise SystemExit(0 if worst < 1e-9 else 1)
