"""Basic joint control of the 4-joint Excavator_Final robot in Isaac Sim 5.1.

Run with:
    ~/isaacsim/python.sh scripts/excavator_control.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Excavator_Final_description", "urdf", "Excavator_Final", "Excavator_Final.usd",
)
PRIM_PATH = "/World/Excavator"

if not os.path.isfile(USD_PATH):
    simulation_app.close()
    raise SystemExit(
        f"USD not found: {USD_PATH}\n"
        "Import Excavator_Final_description/urdf/Excavator_Final.urdf in Isaac Sim first "
        "(Fixed base, Position drive)."
    )

# All joints are 'continuous' in the URDF, so these ranges (radians) are what keeps
# the motion sensible. Each joint sweeps lo -> hi -> lo with the given angular frequency.
JOINT_MOTION = {
    # name:          (lo,    hi,   freq)
    "swing_joint":  (0.0,   np.pi, 0.3),
    "arm_joint":    (0.0,   0.8,   0.5),
    "pole_joint":   (-0.4,  0.4,   0.7),
    "bucket_joint": (-0.5,  0.5,   0.9),
}

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)

excavator = world.scene.add(SingleArticulation(prim_path=PRIM_PATH, name="excavator"))
world.reset()

print("DOF names:", excavator.dof_names)
joint_idx = {name: excavator.get_dof_index(name) for name in JOINT_MOTION}

# Make sure the position drives are stiff enough to track targets.
controller = excavator.get_articulation_controller()
controller.set_gains(
    kps=np.full(excavator.num_dof, 1e4),
    kds=np.full(excavator.num_dof, 1e3),
)

t = 0.0
dt = world.get_physics_dt()
step = 0
while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    t += dt
    step += 1
    targets = excavator.get_joint_positions()
    for name, (lo, hi, freq) in JOINT_MOTION.items():
        # Smooth back-and-forth motion between lo and hi.
        targets[joint_idx[name]] = lo + (hi - lo) * 0.5 * (1 - np.cos(freq * t))
    excavator.apply_action(ArticulationAction(joint_positions=targets))

    if step % 60 == 0:
        pos = excavator.get_joint_positions()
        print(f"t={t:5.1f}s  " + "  ".join(
            f"{name.removesuffix('_joint')}={pos[i]:+.2f}" for name, i in joint_idx.items()
        ))

simulation_app.close()
