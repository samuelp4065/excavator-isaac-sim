"""Regenerate Excavator_Final.usd from Excavator_Final.urdf (headless, no GUI).

Same settings as File -> Import with the defaults: Fixed base, Position drive.
Run with Isaac Sim closed:
    ~/isaacsim/python.sh scripts/import_urdf.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os

import omni.kit.commands
from pxr import Usd

URDF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Excavator_Final_description", "urdf"
)
URDF_PATH = os.path.normpath(os.path.join(URDF_DIR, "Excavator_Final.urdf"))
USD_PATH = os.path.normpath(os.path.join(URDF_DIR, "Excavator_Final", "Excavator_Final.usd"))

_, config = omni.kit.commands.execute("URDFCreateImportConfig")
config.set_merge_fixed_joints(True)
config.set_fix_base(True)
config.set_self_collision(False)
config.set_import_inertia_tensor(True)
config.set_distance_scale(1.0)
config.set_default_drive_type(1)  # 1 = position
config.set_default_drive_strength(1e3)
config.set_default_position_drive_damping(1e2)
config.set_up_vector(0, 0, 1)
config.set_make_default_prim(True)
config.set_create_physics_scene(True)
config.set_collision_from_visuals(False)

result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=URDF_PATH, import_config=config, dest_path=USD_PATH
)
print(f"Import result={result} prim={prim_path}")

# Sanity check: the file must have a default prim with the four joints in it.
stage = Usd.Stage.Open(USD_PATH)
default_prim = stage.GetDefaultPrim()
joints = [p.GetName() for p in stage.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint"]
print(f"Wrote {USD_PATH}")
print(f"  defaultPrim: {default_prim.GetPath() if default_prim else None}")
print(f"  revolute joints: {joints}")

simulation_app.close()
