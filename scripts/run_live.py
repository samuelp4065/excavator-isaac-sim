"""Start Isaac Sim once and hot-reload scripts/live_controller.py on every save.

Run with:
    ~/isaacsim/python.sh scripts/run_live.py

Then edit live_controller.py and save; changes apply within ~0.5 s, no restart.
Only changes to *this* file (scene setup) need a restart.
Click inside the viewport first so keyboard input reaches the sim.
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import importlib.util
import os
import traceback
from types import SimpleNamespace

import carb.input
import numpy as np
import omni.appwindow
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLER_PATH = os.path.join(SCRIPT_DIR, "live_controller.py")
USD_PATH = os.path.join(
    SCRIPT_DIR, "..", "Excavator_Final_description", "urdf", "Excavator_Final", "Excavator_Final.usd",
)
PRIM_PATH = "/World/Excavator"
RELOAD_CHECK_EVERY = 30  # physics steps between file mtime checks (~0.5 s at 60 Hz)

if not os.path.isfile(USD_PATH):
    simulation_app.close()
    raise SystemExit(
        f"USD not found: {USD_PATH}\n"
        "Import Excavator_Final_description/urdf/Excavator_Final.urdf in Isaac Sim first "
        "(Fixed base, Position drive)."
    )

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)

excavator = world.scene.add(SingleArticulation(prim_path=PRIM_PATH, name="excavator"))
world.reset()

print("DOF names:", excavator.dof_names)
controller = excavator.get_articulation_controller()

# Track which keys are currently held down; the controller reads this set.
held = set()


def on_keyboard_event(event, *args):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        held.add(event.input)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held.discard(event.input)
    return True


input_iface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)


def load_controller():
    """Import live_controller.py fresh; returns None (and prints why) on error."""
    spec = importlib.util.spec_from_file_location("live_controller", CONTROLLER_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        controller.set_gains(
            kps=np.full(excavator.num_dof, float(getattr(module, "KP", 1e4))),
            kds=np.full(excavator.num_dof, float(getattr(module, "KD", 1e3))),
        )
    except Exception:
        traceback.print_exc()
        return None
    return module


# State that survives reloads, so the robot doesn't jump when you save.
ctx = SimpleNamespace(
    t=0.0,
    dt=world.get_physics_dt(),
    step=0,
    targets=excavator.get_joint_positions().copy(),
    positions=excavator.get_joint_positions(),
    idx={name: i for i, name in enumerate(excavator.dof_names)},
    held=held,
    excavator=excavator,
)

module = load_controller()
mtime = os.path.getmtime(CONTROLLER_PATH)
print(f"Loaded {CONTROLLER_PATH}" if module else "Controller failed to load; fix and save.")
step_failed = False

while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    ctx.step += 1
    ctx.t += ctx.dt

    if ctx.step % RELOAD_CHECK_EVERY == 0:
        try:
            new_mtime = os.path.getmtime(CONTROLLER_PATH)
        except OSError:  # editors may briefly remove the file while saving
            new_mtime = mtime
        if new_mtime != mtime:
            mtime = new_mtime
            new_module = load_controller()
            if new_module:
                module, step_failed = new_module, False
                print(f"Reloaded live_controller.py (mode={getattr(module, 'MODE', '?')})")
            else:
                print("Reload failed; still running the previous version.")

    if module and not step_failed:
        ctx.positions = excavator.get_joint_positions()
        try:
            module.step(ctx)
        except Exception:
            traceback.print_exc()
            print("step() raised; paused control until live_controller.py is saved again.")
            step_failed = True

    excavator.apply_action(ArticulationAction(joint_positions=ctx.targets))

input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
simulation_app.close()
