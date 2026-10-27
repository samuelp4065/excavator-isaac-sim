"""Keyboard joint control of the 4-joint Excavator_Final robot in Isaac Sim 5.1.

Click inside the viewport so it has keyboard focus, then hold:
    Left / Right   swing   (swing_joint)
    Q / A          boom    (arm_joint)
    W / S          stick   (pole_joint)
    E / D          bucket  (bucket_joint)
    R              reset all joints to 0
Current angles are printed (in degrees) whenever a joint moves.

Run with:
    ~/isaacsim/python.sh scripts/excavator_keyboard.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os

import carb.input
import numpy as np
import omni.appwindow
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

K = carb.input.KeyboardInput
# All joints are 'continuous' in the URDF, so these clamps (radians) are the only
# bound on motion.
JOINT_KEYS = {
    # name:          (key +,   key -,  lo,     hi)
    "swing_joint":  (K.LEFT,  K.RIGHT, -np.pi, np.pi),
    # Collision-free ranges (kinematics.JOINT_LIMITS): beyond them a link swings
    # through its neighbour, since the sim doesn't collide neighbouring links.
    "arm_joint":    (K.Q,     K.A,     np.radians(-50), np.radians(110)),
    "pole_joint":   (K.W,     K.S,     np.radians(-5),  np.radians(130)),
    "bucket_joint": (K.E,     K.D,     np.radians(-8),  np.radians(278)),  # hits the stick at -80..-10 deg
}
RESET_KEY = K.R
SPEED = 0.5  # rad/s while a key is held (~29 deg/s)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)

excavator = world.scene.add(SingleArticulation(prim_path=PRIM_PATH, name="excavator"))
world.reset()

print("DOF names:", excavator.dof_names)
joint_idx = {name: excavator.get_dof_index(name) for name in JOINT_KEYS}

controller = excavator.get_articulation_controller()
controller.set_gains(
    kps=np.full(excavator.num_dof, 1e4),
    kds=np.full(excavator.num_dof, 1e3),
)

# Track which keys are currently held down.
held = set()
reset_requested = False


def on_keyboard_event(event, *args):
    global reset_requested
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        held.add(event.input)
        if event.input == RESET_KEY:
            reset_requested = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held.discard(event.input)
    return True


input_iface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

print(__doc__.split("Run with:")[0])

# Commanded targets; kept separately from measured positions so they don't drift.
targets = excavator.get_joint_positions().copy()
dt = world.get_physics_dt()
step = 0
while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    step += 1
    moved = False
    if reset_requested:
        targets[:] = 0.0
        reset_requested = False
        moved = True
    for name, (key_up, key_down, lo, hi) in JOINT_KEYS.items():
        direction = (key_up in held) - (key_down in held)
        if direction:
            i = joint_idx[name]
            targets[i] = np.clip(targets[i] + direction * SPEED * dt, lo, hi)
            moved = True
    excavator.apply_action(ArticulationAction(joint_positions=targets))

    if moved and step % 6 == 0:
        pos = excavator.get_joint_positions()
        print("  ".join(
            f"{name.removesuffix('_joint')}={np.rad2deg(pos[i]):+7.1f}°"
            f" (cmd {np.rad2deg(targets[i]):+7.1f}°)"
            for name, i in joint_idx.items()
        ))

input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
simulation_app.close()
