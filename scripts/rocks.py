"""Rock pile and dump bin for run_live.py.

Hot-reloaded: save this file while run_live.py is running and the rocks and bin
are rebuilt in place (old ones removed, new pile poured and settled, excavator
kept where it is). No Isaac Sim restart needed. T (in live_controller.py) puts
the rocks back into the settled pile without rebuilding.

Each rock is a randomly lumped low-poly ball with a convex-hull collider, so they
interlock and hold a heap shape like real gravel, and stay cheap to simulate.
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

# --- Rocks ------------------------------------------------------------------
NUM_ROCKS = 150
ROCK_SIZE = 0.02           # m, typical diameter (bucket is ~13 cm wide)
SIZE_JITTER = 0.25         # +/- fraction
LUMPINESS = 0.25           # how far surface points stray from a sphere
ANGULAR = False            # True: 12-corner chunky stones that don't roll; False: rounder
SEED = 0                   # change for a different set of rock shapes

# --- Pile -------------------------------------------------------------------
# Loose rocks slump flat, so by default they're poured into a low-walled dig box
# that keeps them heaped several layers deep. Without the box they're stacked
# into a cone and left to slump.
DIG_BOX = True
DIG_BOX_SIZE = 0.16        # m, inside width (bucket is ~13 cm)
DIG_BOX_WALL_HEIGHT = 0.04 # m
PILE_SLOPE_DEG = 35.0      # cone slope when DIG_BOX is False
# Centre of the pile: this far from the swing axis, in the direction the boom
# points at swing=0 (where the bucket can reach).
PILE_DISTANCE = 0.45       # m
SETTLE_SECONDS = 4.0       # max sim time spent settling the pour

# --- Dump bin ---------------------------------------------------------------
DUMP_BIN = True
BIN_SWING_DEG = 90.0       # where the bin sits, as a swing angle (+ = counter-clockwise)
BIN_DISTANCE = 0.45        # m from the swing axis
BIN_SIZE = 0.18            # m, inside width
BIN_WALL_HEIGHT = 0.05     # m

# --- Material ---------------------------------------------------------------
STATIC_FRICTION = 0.8
DYNAMIC_FRICTION = 0.7
RESTITUTION = 0.1
DENSITY = 2600.0           # kg/m^3, granite-ish
ANGULAR_DAMPING = 0.0      # >0 makes rocks stop rolling sooner

ROOT = "/World/Rocks"
SWING_AXIS_XY = np.array([0.0, 0.0375])
BOOM_DIR_XY = np.array([0.151061, -0.988524])  # horizontal boom direction at swing=0


def _rotate_xy(v, deg):
    a = np.radians(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def _icosphere(subdivide=True):
    """Unit icosphere: 12 vertices / 20 triangles, or subdivided once (42 / 80)."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    if not subdivide:
        return np.array(verts), np.array(faces)
    midpoints = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoints:
            m = verts[a] + verts[b]
            verts.append(m / np.linalg.norm(m))
            midpoints[key] = len(verts) - 1
        return midpoints[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts), np.array(new_faces)


def _rock_points(rng, size):
    verts, faces = _icosphere(subdivide=not ANGULAR)
    radial = 1 + rng.uniform(-LUMPINESS, LUMPINESS, len(verts))
    squash = rng.uniform(0.65, 1.0, 3)  # flatten/elongate a little so they aren't balls
    pts = verts * radial[:, None] * squash * (size / 2)
    return pts, faces


def _cone(radius, spacing, centre):
    shrink = spacing / np.tan(np.radians(PILE_SLOPE_DEG))  # radius lost per layer
    pts, k = [], 0
    while radius - k * shrink > 0:
        r_layer = radius - k * shrink
        off = spacing / 2 if k % 2 else 0.0  # stagger layers
        n = int(r_layer / spacing) + 1
        z = k * spacing
        pts += [(centre[0] + i * spacing + off, centre[1] + j * spacing + off, z)
                for i in range(-n, n + 1) for j in range(-n, n + 1)
                if (i * spacing + off) ** 2 + (j * spacing + off) ** 2 <= r_layer ** 2]
        k += 1
    return pts


def _pile_positions(rng, radii):
    """Non-overlapping spots stacked in a cone in front of the excavator: the
    smallest cone that holds every rock, with any extras trimmed off its rim."""
    spacing = 2 * radii.max() * 1.05
    centre = SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY
    if DIG_BOX:
        # Column filling the box footprint; it settles into the box.
        n = int((DIG_BOX_SIZE / 2 - radii.max()) / spacing)
        side = np.array([-BOOM_DIR_XY[1], BOOM_DIR_XY[0]])
        layer = [centre + i * spacing * side + j * spacing * BOOM_DIR_XY
                 for i in range(-n, n + 1) for j in range(-n, n + 1)]
        pts = np.array([(x, y, k * spacing) for k in range(len(radii) // len(layer) + 1)
                        for x, y in layer])[:len(radii)]
        pts[:, 2] += 0.01 + radii.max() + 0.002  # above the box floor
        return pts
    radius = spacing
    while len(pts := _cone(radius, spacing, centre)) < len(radii):
        radius *= 1.05
    pts = np.array(pts)
    rim = np.linalg.norm(pts[:, :2] - centre, axis=1)
    pts = pts[np.argsort(rim)[:len(radii)]]
    pts[:, 2] += radii.max() + 0.002
    return pts


def _build_box(stage, path, centre, dir_xy, inner, height, color):
    """Open-top box of static colliders (no rigid body, so it never moves)."""
    yaw = np.degrees(np.arctan2(dir_xy[1], dir_xy[0]))
    t, h, w = 0.01, height, inner
    parts = {  # name: (local offset, size)
        "floor": ((0, 0, t / 2), (w + 2 * t, w + 2 * t, t)),
        "wall_n": ((0, (w + t) / 2, h / 2), (w + 2 * t, t, h)),
        "wall_s": ((0, -(w + t) / 2, h / 2), (w + 2 * t, t, h)),
        "wall_e": (((w + t) / 2, 0, h / 2), (t, w, h)),
        "wall_w": ((-(w + t) / 2, 0, h / 2), (t, w, h)),
    }
    root = UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.XformCommonAPI(root)
    xf.SetTranslate(Gf.Vec3d(centre[0], centre[1], 0))
    xf.SetRotate(Gf.Vec3f(0, 0, float(yaw)))
    for name, (off, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"{path}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cxf = UsdGeom.XformCommonAPI(cube)
        cxf.SetTranslate(Gf.Vec3d(*off))
        cxf.SetScale(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


class RockSet:
    def __init__(self, stage):
        rng = np.random.default_rng(SEED)
        UsdGeom.Scope.Define(stage, ROOT)

        mat_path = ROOT + "/RockMaterial"
        UsdShade.Material.Define(stage, mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        phys_mat.CreateStaticFrictionAttr(STATIC_FRICTION)
        phys_mat.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
        phys_mat.CreateRestitutionAttr(RESTITUTION)
        phys_mat.CreateDensityAttr(DENSITY)
        material = UsdShade.Material(stage.GetPrimAtPath(mat_path))

        shapes = [_rock_points(rng, ROCK_SIZE * (1 + rng.uniform(-SIZE_JITTER, SIZE_JITTER)))
                  for _ in range(NUM_ROCKS)]
        radii = np.array([np.linalg.norm(pts, axis=1).max() for pts, _ in shapes])
        positions = _pile_positions(rng, radii)

        self.count = NUM_ROCKS
        for i, ((pts, faces), pos) in enumerate(zip(shapes, positions)):
            mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/rock{i}")
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")  # keep the facets: looks rockier
            grey = rng.uniform(0.35, 0.55)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(grey, grey * 0.95, grey * 0.9)])

            xf = UsdGeom.XformCommonAPI(mesh)
            xf.SetTranslate(Gf.Vec3d(*pos))
            xf.SetRotate(Gf.Vec3f(*rng.uniform(0, 360, 3)))

            prim = mesh.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateAngularDampingAttr(ANGULAR_DAMPING)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
        if DIG_BOX:
            _build_box(stage, ROOT + "/DigBox", SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY,
                       BOOM_DIR_XY, DIG_BOX_SIZE, DIG_BOX_WALL_HEIGHT, (0.45, 0.33, 0.2))
        if DUMP_BIN:
            bin_dir = _rotate_xy(BOOM_DIR_XY, BIN_SWING_DEG)
            _build_box(stage, ROOT + "/Bin", SWING_AXIS_XY + BIN_DISTANCE * bin_dir,
                       bin_dir, BIN_SIZE, BIN_WALL_HEIGHT, (0.25, 0.35, 0.55))
        self.view = None
        self.initial_positions = positions
        self.initial_orientations = None
        print(f"[rocks] {NUM_ROCKS} rocks, ~{ROCK_SIZE * 100:.1f} cm each"
              + (f", dump bin at swing {BIN_SWING_DEG:+.0f} deg" if DUMP_BIN else ""))

    def initialize(self):
        """Call after world.reset(): one batched view over all rocks."""
        from isaacsim.core.prims import RigidPrim
        self.view = RigidPrim(f"{ROOT}/rock.*", name="rocks_view")
        self.view.initialize()
        self.initial_positions, self.initial_orientations = self.view.get_world_poses()

    def settle(self, world, before_step=None):
        """Settle the pile: step physics (no rendering) until the rocks stop moving,
        then remember that state for reset(). before_step() runs each step, e.g. to
        keep the excavator holding its pose."""
        dt = world.get_physics_dt()
        steps = 0
        prev = self.view.get_world_poses()[0]
        while steps * dt < SETTLE_SECONDS:
            for _ in range(30):
                if before_step:
                    before_step()
                world.step(render=False)
            steps += 30
            pos = self.view.get_world_poses()[0]
            # Judge rest by actual movement; reported velocities carry a small offset.
            moved = np.linalg.norm(pos - prev, axis=1)
            prev = pos
            if steps * dt > 0.5 and np.percentile(moved, 95) < 0.0005:
                break
        self.initial_positions, self.initial_orientations = self.view.get_world_poses()
        print(f"[rocks] pile settled in {steps * dt:.1f} s sim time, "
              f"height {self.initial_positions[:, 2].max() * 100:.1f} cm")

    def reset(self):
        """Put every rock back into the settled pile."""
        if self.view is None:
            return
        self.view.set_world_poses(positions=self.initial_positions, orientations=self.initial_orientations)
        self.view.set_velocities(np.zeros((self.count, 6)))
        print("[rocks] reset")

    def positions(self):
        """(N, 3) array of current rock centres, e.g. to count rocks in the bucket."""
        return self.view.get_world_poses()[0] if self.view is not None else self.initial_positions
