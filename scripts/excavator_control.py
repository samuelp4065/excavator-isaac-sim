"""Basic joint control of the Excavator_Final robot in Isaac Sim 5.1.

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

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)

excavator = world.scene.add(SingleArticulation(prim_path=PRIM_PATH, name="excavator"))
world.reset()

print("DOF names:", excavator.dof_names)
swing_idx = excavator.get_dof_index("swing_joint")
arm_idx = excavator.get_dof_index("arm_joint")

# Make sure the position drives are stiff enough to track targets.
controller = excavator.get_articulation_controller()
controller.set_gains(
    kps=np.full(excavator.num_dof, 1e4),
    kds=np.full(excavator.num_dof, 1e3),
)

# Joint limits from the URDF (radians).
SWING_MIN, SWING_MAX = 0.0, 3.141593
ARM_MIN, ARM_MAX = 0.0, 0.959931

t = 0.0
dt = world.get_physics_dt()
while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    t += dt
    # Smooth back-and-forth motion within each joint's limits.
    swing_target = SWING_MIN + (SWING_MAX - SWING_MIN) * 0.5 * (1 - np.cos(0.3 * t))
    arm_target = ARM_MIN + (ARM_MAX - ARM_MIN) * 0.5 * (1 - np.cos(0.6 * t))

    targets = excavator.get_joint_positions()
    targets[swing_idx] = swing_target
    targets[arm_idx] = arm_target
    excavator.apply_action(ArticulationAction(joint_positions=targets))

    if int(t / dt) % 60 == 0:
        print(f"t={t:5.1f}s  swing={excavator.get_joint_positions()[swing_idx]:.2f}  "
              f"arm={excavator.get_joint_positions()[arm_idx]:.2f}")

simulation_app.close()
