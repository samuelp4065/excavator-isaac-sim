"""Check kinematics.py against the simulator (headless, no rocks).

Run with Isaac Sim closed:
    ~/isaacsim/python.sh scripts/check_kinematics.py

1. FK: put the robot in random poses, compare kinematics.fk() of the joint
   angles the sim actually reports with where the bucket's cutting edge is.
   This tests the model alone.
2. IK: pick random tip targets, solve with kinematics.ik(), pose the robot
   there and measure how far the cutting edge is from the target.
3. Tracking (info only, not pass/fail): command the IK pose through the joint
   drives and let it settle; the extra error is the controller (drive
   stiffness, gravity sag), not the kinematics.
Prints PASS/FAIL and writes scripts/check_kinematics.txt. (The exit code isn't
reliable: Isaac Sim's shutdown replaces it.)
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import sys

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim, SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as K

USD_PATH = os.path.join(K.REPO, "Excavator_Final_description", "urdf", "Excavator_Final",
                        "Excavator_Final.usd")
N_POSES = 200
FK_TOLERANCE = 0.001  # m, model only
IK_TOLERANCE = 0.003  # m (includes the arm sagging a little during the posing step)

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/Excavator")
excavator = world.scene.add(SingleArticulation("/World/Excavator", name="excavator"))
world.reset()
# Same stiff drives as the controllers use, so commanded poses are held.
excavator.get_articulation_controller().set_gains(kps=np.full(4, 1e4), kds=np.full(4, 1e3))
bucket = RigidPrim("/World/Excavator/Bucket_1", name="bucket_link")
bucket.initialize()
names = excavator.dof_names


def bucket_pose():
    p, q = bucket.get_world_poses()  # q is (w, x, y, z)
    return p[0], Rotation.from_quat([q[0][1], q[0][2], q[0][3], q[0][0]])


def pose_robot(q, settle_steps=1):
    """Teleport the joints there, make it the drive target too, and step physics
    so the link poses update. Returns the joint angles the sim then reports."""
    a = K.as_array(K.as_dict(q), names)
    excavator.set_joint_positions(a)
    excavator.set_joint_velocities(np.zeros(len(a)))
    for _ in range(settle_steps):
        excavator.apply_action(ArticulationAction(joint_positions=a))
        world.step(render=False)
    return dict(zip(names, excavator.get_joint_positions()))


# The cutting edge is a fixed point on the bucket link. Record where it is in a
# reference pose (from FK of the angles the sim actually reached, since the arm
# sags slightly even in one step), then follow it through the link's motion.
TIP_REF = K.fk(pose_robot(np.zeros(4)))[0]
P0, R0 = bucket_pose()


def sim_tip():
    P, R = bucket_pose()
    return (R * R0.inv()).apply(TIP_REF - P0) + P


rng = np.random.default_rng(0)

# 1. Forward kinematics
fk_err = []
for _ in range(N_POSES):
    q = dict(zip(K.JOINTS, rng.uniform(-1.5, 1.5, 4)))
    measured = pose_robot(q)
    fk_err.append(np.linalg.norm(K.fk(measured)[0] - sim_tip()))
fk_err = np.array(fk_err)

# 2. Inverse kinematics: targets made from random reachable poses
ik_err, sag, track_err, unreachable = [], [], [], 0
for _ in range(N_POSES):
    target, pitch = K.fk(dict(zip(K.JOINTS, rng.uniform(-1.2, 1.2, 4))))
    try:
        q = K.ik(target, pitch)
    except ValueError:
        unreachable += 1
        continue
    measured = pose_robot(q)
    ik_err.append(np.linalg.norm(sim_tip() - target))
    sag.append(np.linalg.norm(K.fk(measured)[0] - target))  # tip moved by the one physics step
    pose_robot(q, settle_steps=60)
    track_err.append(np.linalg.norm(sim_tip() - target))
ik_err, sag, track_err = np.array(ik_err), np.array(sag), np.array(track_err)

lines = [
    K.describe(),
    f"FK vs sim, {len(fk_err)} poses: mean {fk_err.mean() * 1000:.2f} mm, max {fk_err.max() * 1000:.2f} mm",
    f"IK -> sim, {len(ik_err)} targets: mean {ik_err.mean() * 1000:.2f} mm, max {ik_err.max() * 1000:.2f} mm"
    + (f" ({unreachable} unreachable skipped)" if unreachable else "")
    + f"; of which sag during the posing step: mean {sag.mean() * 1000:.2f} mm",
    f"Drive tracking after 1 s (info): mean {track_err.mean() * 1000:.1f} mm, max {track_err.max() * 1000:.1f} mm",
]
ok = fk_err.max() < FK_TOLERANCE and ik_err.max() < IK_TOLERANCE
lines.append("PASS" if ok else
             f"FAIL (tolerances: FK {FK_TOLERANCE * 1000:.0f} mm, IK {IK_TOLERANCE * 1000:.0f} mm)")
# Kit's log handling can swallow stdout in headless runs, so also write a file.
report = os.path.join(K.REPO, "scripts", "check_kinematics.txt")
with open(report, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

simulation_app.close()
