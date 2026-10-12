"""Start Isaac Sim once and hot-reload the controller and the rock scene on save.

Run with:
    ~/isaacsim/python.sh scripts/run_live.py

Then, while it runs:
- save live_controller.py -> control logic reloads within ~0.5 s;
- save rocks.py -> rocks and dump bin are rebuilt and the pile re-poured
  (a second or two), the excavator keeps its pose.
Only changes to *this* file or a re-imported robot USD need a restart.
Click inside the viewport first so keyboard input reaches the sim.
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import traceback
import types
from types import SimpleNamespace

import carb.input
import numpy as np
import omni.appwindow
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
CONTROLLER_PATH = os.path.join(SCRIPT_DIR, "live_controller.py")
ROCKS_PATH = os.path.join(SCRIPT_DIR, "rocks.py")
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
# A handful of bodies runs fastest on the CPU solver (GPU dynamics only pays off
# with thousands). Fabric keeps physics results out of USD, so updates are cheap.
world.get_physics_context().enable_fabric(True)

world.scene.add_default_ground_plane(static_friction=0.8, dynamic_friction=0.7)
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


def load_module(path, name):
    """Run a file fresh (picking up edits); returns None (and prints why) on error.

    Compiles the source directly instead of importing: Isaac Sim's Python can keep
    serving the stale cached bytecode from startup, silently ignoring edits."""
    module = types.ModuleType(name)
    module.__file__ = path
    try:
        with open(path) as f:
            code = compile(f.read(), path, "exec")
        exec(code, module.__dict__)
    except Exception:
        traceback.print_exc()
        return None
    return module


def apply_gains(mod):
    controller.set_gains(
        kps=np.full(excavator.num_dof, float(getattr(mod, "KP", 1e4))),
        kds=np.full(excavator.num_dof, float(getattr(mod, "KD", 1e3))),
    )


def load_controller():
    new = load_module(CONTROLLER_PATH, "live_controller")
    if new is None:
        return None
    try:
        apply_gains(new)
    except Exception:
        traceback.print_exc()
        return None
    return new


def hold_pose():
    excavator.apply_action(ArticulationAction(joint_positions=ctx.targets))


def rebuild_rocks():
    """(Re)build the rock scene from rocks.py without restarting Isaac Sim."""
    new = load_module(ROCKS_PATH, "rocks")
    if new is None:
        print("rocks.py failed to load; keeping the current rocks.")
        return
    # Physics views must be rebuilt after rigid bodies are added or removed, so
    # stop, swap the prims, and start again with the excavator in the same pose.
    world.stop()
    for root in {getattr(ctx.rocks_module, "ROOT", None), new.ROOT} - {None}:
        if world.stage.GetPrimAtPath(root):
            world.stage.RemovePrim(root)
    try:
        rocks = new.RockSet(world.stage)
    except Exception:
        traceback.print_exc()
        rocks = None
        print("Building rocks failed; fix rocks.py and save again.")
    world.reset()
    excavator.set_joint_positions(ctx.targets)
    excavator.set_joint_velocities(np.zeros(excavator.num_dof))
    if module:
        apply_gains(module)
    if rocks:
        try:
            rocks.initialize()
            rocks.settle(world, before_step=hold_pose)
        except Exception:
            traceback.print_exc()
    ctx.rocks, ctx.rocks_module = rocks, new


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
    rocks=None,
    rocks_module=None,
)

module = load_controller()
print(f"Loaded {CONTROLLER_PATH}" if module else "Controller failed to load; fix and save.")
rebuild_rocks()
step_failed = False


def mtime_of(path, last):
    try:
        return os.path.getmtime(path)
    except OSError:  # editors may briefly remove the file while saving
        return last


mtime = mtime_of(CONTROLLER_PATH, 0)
rocks_mtime = mtime_of(ROCKS_PATH, 0)

while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    ctx.step += 1
    ctx.t += ctx.dt

    if ctx.step % RELOAD_CHECK_EVERY == 0:
        new_mtime = mtime_of(ROCKS_PATH, rocks_mtime)
        if new_mtime != rocks_mtime:
            rocks_mtime = new_mtime
            print("rocks.py changed; rebuilding rocks...")
            rebuild_rocks()
        new_mtime = mtime_of(CONTROLLER_PATH, mtime)
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
