"""Rock pile and dump bin for run_live.py.

Hot-reloaded: save this file while run_live.py is running and the rocks and bin
are rebuilt in place (old ones removed, new pile poured and settled, excavator
kept where it is). No Isaac Sim restart needed. T (in live_controller.py) puts
the rocks back into the settled pile without rebuilding.

Each rock is a randomly lumped low-poly ball with a convex-hull collider, so they
interlock and hold a heap shape like real gravel, and stay cheap to simulate.
"""
import numpy as np

try:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
except ImportError:  # plain Python (e.g. getpose.py): settings and terrain_height() still work
    pass

# --- Rocks ------------------------------------------------------------------
NUM_ROCKS = 900
ROCK_TYPE = "limestone"    # a preset from ROCK_TYPES
SEED = 0                   # change for a different set of rock shapes
ROCK_TYPES = {
    # size: typical diameter (m, bucket is ~13 cm wide); jitter: +/- size fraction;
    # lumpiness: how irregular (0 = neat blocks); flatness: thickness range as a
    # fraction of the length; colour: RGB tint scaled by a random brightness.
    "granite":   dict(size=0.02, jitter=0.25, lumpiness=0.25, flatness=(0.4, 0.7),
                      colour=(1.0, 0.95, 0.9), brightness=(0.35, 0.55),
                      density=2600.0, static_friction=0.8, dynamic_friction=0.7),
    # Crushed limestone: pale cream-grey, flat angular slabs, a bit lighter,
    # rougher faces.
    "limestone": dict(size=0.025, jitter=0.35, lumpiness=0.3, flatness=(0.35, 0.6),
                      colour=(1.0, 0.95, 0.83), brightness=(0.62, 0.85),
                      density=2500.0, static_friction=0.9, dynamic_friction=0.8),
}
_T = ROCK_TYPES[ROCK_TYPE]
ROCK_SIZE, SIZE_JITTER, LUMPINESS, FLATNESS = _T["size"], _T["jitter"], _T["lumpiness"], _T["flatness"]

# --- Pile -------------------------------------------------------------------
# Loose rocks alone slump to a single layer in PhysX, so the pile needs support:
#   "mound": a solid cone of dirt covered in a few layers of rocks. The bucket
#            scrapes rocks off the slope without having to dig below ground.
#   "crater": an upside-down cone: a raised ring of dirt whose inner wall slopes
#            down to ground level. Rocks slide to the middle and heap up deep, and
#            the bucket never has to go below ground (the ground plane is solid,
#            so the crater can't be sunk into it).
#   "box":   rocks poured into a low-walled dig box, heaped a few layers deep.
#   "pit":   a hole sunk below ground level, rocks poured in. The pit walls hold
#            the rocks while the bucket's lip passes under them, as in real digging
#            (on flat ground the lip just bulldozes rocks along). The ground plane
#            is solid, so the surface here is a static dirt slab with the hole cut
#            out and the real ground plane lowered to GROUND_PLANE_Z under it.
#            Switching to/from "pit" moves the ground plane: restart run_live.py.
PILE_STYLE = "pit"
PIT_RIM_RADIUS = 0.21      # m, hole radius at ground level
PIT_FLOOR_RADIUS = 0.11    # m, flat bottom (0.15 m deep over 0.10 m = 56 degree wall)
PIT_DEPTH = 0.15           # m below ground (900 limestone rocks fill it to near the rim)
SLAB_RADIUS = 2.5          # m, extent of the dirt slab around the pit
GROUND_PLANE_Z = -0.25 if PILE_STYLE == "pit" else 0.0  # read by run_live.py / pick_rock.py
CRATER_RIM_RADIUS = 0.15   # m, top of the inner wall
CRATER_FLOOR_RADIUS = 0.05 # m, flat ground-level floor in the middle
CRATER_DEPTH = 0.06        # m, rim height (0.06 over 0.10 m = 31 degree wall)
CRATER_OUTER_RADIUS = 0.22 # m, where the outside of the ring meets the ground
MOUND_RADIUS = 0.15        # m, base of the dirt cone
MOUND_HEIGHT = 0.07        # m (0.07 / 0.15 = 25 degree slope)
DIG_BOX_SIZE = 0.16        # m, inside width (bucket is ~13 cm)
DIG_BOX_WALL_HEIGHT = 0.04 # m
# Centre of the pile: this far from the swing axis, in the direction the boom
# points at swing=0 (where the bucket can reach).
PILE_DISTANCE = 0.40 if PILE_STYLE == "pit" else 0.45  # m (near rim clear of the base, r 0.14 m; dig reach ends ~0.53 m)
SETTLE_SECONDS = 6.0       # max sim time spent settling the pour

# --- Dump bin ---------------------------------------------------------------
DUMP_BIN = True
BIN_SWING_DEG = 90.0       # where the bin sits, as a swing angle (+ = counter-clockwise)
BIN_DISTANCE = 0.45        # m from the swing axis
BIN_SIZE = 0.18            # m, inside width
BIN_WALL_HEIGHT = 0.05     # m

# --- Material ---------------------------------------------------------------
STATIC_FRICTION = _T["static_friction"]
DYNAMIC_FRICTION = _T["dynamic_friction"]
RESTITUTION = 0.1
DENSITY = _T["density"]    # kg/m^3
ANGULAR_DAMPING = 0.0      # >0 makes rocks stop rolling sooner

ROOT = "/World/Rocks"
SWING_AXIS_XY = np.array([0.0, 0.0375])
BOOM_DIR_XY = np.array([0.151061, -0.988524])  # horizontal boom direction at swing=0


def _rotate_xy(v, deg):
    a = np.radians(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def _rock_points(rng, size):
    """Irregular chunky block: jittered box corners plus a few extra surface
    points, wrapped in their convex hull. Flat faces matter: round rocks roll
    down any slope however high the friction, so piles can't form."""
    from scipy.spatial import ConvexHull
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners *= 1 + rng.uniform(-LUMPINESS, LUMPINESS, corners.shape)
    extra = rng.uniform(-1, 1, (4, 3))
    extra[np.arange(4), rng.integers(0, 3, 4)] = rng.choice([-1, 1], 4)  # push onto a face
    pts = np.vstack([corners, extra]) * np.array([1, rng.uniform(0.6, 1), rng.uniform(*FLATNESS)]) * (size / 2)
    hull = ConvexHull(pts)
    pts = pts[hull.vertices]
    remap = {v: i for i, v in enumerate(hull.vertices)}
    faces = []
    centre = pts.mean(0)
    for tri in hull.simplices:
        a, b, c = (remap[v] for v in tri)
        if np.dot(np.cross(pts[b] - pts[a], pts[c] - pts[a]), pts[a] - centre) < 0:
            b, c = c, b  # wind outward so the faces render
        faces.append((a, b, c))
    return pts, np.array(faces)


def _mound_height(r):
    return np.maximum(MOUND_HEIGHT * (1 - r / MOUND_RADIUS), 0.0)


def _pile_positions(rng, radii, half_heights):
    """Non-overlapping starting spots for the rocks, layered over the mound or
    stacked in a column over the dig box. They settle into place from there."""
    spacing = 2 * radii.max() * 1.05
    centre = SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY
    side = np.array([-BOOM_DIR_XY[1], BOOM_DIR_XY[0]])
    if PILE_STYLE == "mound":
        # Layers following the slope; vertical step is stretched so neighbours on
        # the slope still don't overlap.
        step = spacing / np.cos(np.arctan(MOUND_HEIGHT / MOUND_RADIUS)) * 1.05
        n = int(MOUND_RADIUS / spacing)
        grid = [(i * spacing, j * spacing) for i in range(-n, n + 1) for j in range(-n, n + 1)
                if (i * spacing) ** 2 + (j * spacing) ** 2 <= MOUND_RADIUS ** 2]
        pts, k = [], 0
        while len(pts) < len(radii):
            off = spacing / 2 if k % 2 else 0.0  # stagger layers
            for x, y in grid:
                x, y = x + off, y + off
                xy = centre + x * side + y * BOOM_DIR_XY
                pts.append((xy[0], xy[1], _mound_height(np.hypot(x, y)) + radii.max() + 0.002 + k * step))
            k += 1
        return np.array(pts[:len(radii)])
    if PILE_STYLE == "pit":
        # Fill the hole from its floor up: flat layers, each packed greedily with
        # each rock's own footprint and as wide as the pit at that height. Rocks go
        # in thickest first and each layer is only as tall as its own rocks, so
        # the pour stays low (below the bucket).
        order = np.argsort(-half_heights)
        grid = np.arange(-PIT_RIM_RADIUS, PIT_RIM_RADIUS + 1e-9, 0.004)
        cand = np.array([(x, y) for x in grid for y in grid])
        cand = cand[rng.permutation(len(cand))]
        pts = np.zeros((len(radii), 3))
        k, z = 0, -PIT_DEPTH + half_heights[order[0]] + 0.002
        while k < len(order):
            wall_r = PIT_RIM_RADIUS if z >= 0 else \
                PIT_FLOOR_RADIUS + (PIT_RIM_RADIUS - PIT_FLOOR_RADIUS) * (z - half_heights[order[k]] + PIT_DEPTH) / PIT_DEPTH
            placed, placed_r, layer_hh = [], [], half_heights[order[k]]
            for c in cand:
                if k >= len(order):
                    break
                i = order[k]
                if np.hypot(*c) > wall_r - radii[i] - 0.002:
                    continue
                if placed and (np.linalg.norm(np.array(placed) - c, axis=1) < np.array(placed_r) + radii[i] + 0.002).any():
                    continue
                placed.append(c)
                placed_r.append(radii[i])
                pts[i] = (*(centre + c[0] * side + c[1] * BOOM_DIR_XY), z)
                k += 1
            if k < len(order):  # next layer: this one's top + the next (thinner) rocks' half height
                z += layer_hh + half_heights[order[k]] + 0.002
        return pts
    if PILE_STYLE == "crater":
        # Column over the crater; rocks slide down the walls into the middle.
        # Rocks lie flat, so layers only need to be as tall as the thickest rock;
        # this keeps the column below the bucket's starting height (~34 cm).
        r = CRATER_RIM_RADIUS - radii.max()
        n = int(r / spacing)
        layer = [centre + i * spacing * side + j * spacing * BOOM_DIR_XY
                 for i in range(-n, n + 1) for j in range(-n, n + 1)
                 if (i * spacing) ** 2 + (j * spacing) ** 2 <= r ** 2]
        dz = 2 * half_heights.max() * 1.05
        pts = np.array([(x, y, k * dz) for k in range(len(radii) // len(layer) + 1)
                        for x, y in layer])[:len(radii)]
        pts[:, 2] += CRATER_DEPTH + half_heights.max() + 0.002  # start above the rim
        return pts
    # Box: column filling the box footprint; it settles into the box.
    n = int((DIG_BOX_SIZE / 2 - radii.max()) / spacing)
    layer = [centre + i * spacing * side + j * spacing * BOOM_DIR_XY
             for i in range(-n, n + 1) for j in range(-n, n + 1)]
    pts = np.array([(x, y, k * spacing) for k in range(len(radii) // len(layer) + 1)
                    for x, y in layer])[:len(radii)]
    pts[:, 2] += 0.01 + radii.max() + 0.002  # above the box floor
    return pts


def _build_crater(stage, path, centre, segments=48):
    """Static ring of dirt: inner wall slopes from the rim down to ground level.
    It's concave, so it uses its exact triangle mesh as the collider (fine for
    static geometry; a convex hull would fill the hole in)."""
    profile = [(CRATER_FLOOR_RADIUS, 0.0), (CRATER_RIM_RADIUS, CRATER_DEPTH),
               (CRATER_OUTER_RADIUS, 0.0)]  # inner foot, rim, outer foot
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pts = [(r * np.cos(t), r * np.sin(t), z) for t in a for r, z in profile]
    m = len(profile)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        for k in range(m - 1):
            p0, p1, q0, q1 = i * m + k, i * m + k + 1, j * m + k, j * m + k + 1
            faces += [(p0, q0, q1), (p0, q1, p1)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.31, 0.2)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(centre[0], centre[1], 0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def _build_pit(stage, path, centre, segments=48):
    """Static dirt slab at ground level with a sloped hole sunk into it. Concave,
    so it uses its exact triangle mesh as the collider (fine for static geometry)."""
    profile = [(0.0, -PIT_DEPTH), (PIT_FLOOR_RADIUS, -PIT_DEPTH), (PIT_RIM_RADIUS, 0.0),
               (PIT_RIM_RADIUS + 0.05, 0.0), (SLAB_RADIUS, 0.0)]  # centre, floor edge, rim, apron, edge
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pts = [(0.0, 0.0, -PIT_DEPTH)] + [(r * np.cos(t), r * np.sin(t), z) for t in a for r, z in profile[1:]]
    m = len(profile) - 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((0, 1 + i * m, 1 + j * m))  # floor fan
        for k in range(m - 1):
            p0, p1, q0, q1 = 1 + i * m + k, 1 + i * m + k + 1, 1 + j * m + k, 1 + j * m + k + 1
            faces += [(p0, p1, q1), (p0, q1, q0)]  # wound so normals point up
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.31, 0.2)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(centre[0], centre[1], 0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def terrain_height(xy):
    """Height of the static dirt (crater, mound or pit) at an (n, 2) array of
    points; 0 elsewhere. For planners that must keep the bucket out of the dirt."""
    r = np.linalg.norm(np.atleast_2d(xy) - (SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY), axis=1)
    if PILE_STYLE == "pit":
        return np.where(r < PIT_FLOOR_RADIUS, -PIT_DEPTH,
                        np.where(r < PIT_RIM_RADIUS,
                                 -PIT_DEPTH * (PIT_RIM_RADIUS - r) / (PIT_RIM_RADIUS - PIT_FLOOR_RADIUS), 0.0))
    if PILE_STYLE == "crater":
        f, rim, out, h = CRATER_FLOOR_RADIUS, CRATER_RIM_RADIUS, CRATER_OUTER_RADIUS, CRATER_DEPTH
        return np.where((r > f) & (r < rim), h * (r - f) / (rim - f),
                        np.where((r >= rim) & (r < out), h * (out - r) / (out - rim), 0.0))
    if PILE_STYLE == "mound":
        return _mound_height(r)
    return np.zeros(len(r))


def _build_mound(stage, path, centre, segments=32):
    """Static dirt cone (collider only, never moves)."""
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pts = [(0, 0, MOUND_HEIGHT), (0, 0, 0)] + [(MOUND_RADIUS * np.cos(t), MOUND_RADIUS * np.sin(t), 0) for t in a]
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(0, 2 + i, 2 + j), (1, 2 + j, 2 + i)]  # slope, bottom
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.31, 0.2)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(centre[0], centre[1], 0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


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
        # Rocks start lying flat (spun about the vertical only), so their footprint
        # radius and half-height bound them separately.
        radii = np.array([np.linalg.norm(pts[:, :2], axis=1).max() for pts, _ in shapes])
        half_heights = np.array([np.abs(pts[:, 2]).max() for pts, _ in shapes])
        positions = _pile_positions(rng, np.maximum(radii, half_heights), half_heights)

        self.count = NUM_ROCKS
        for i, ((pts, faces), pos) in enumerate(zip(shapes, positions)):
            mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/rock{i}")
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")  # keep the facets: looks rockier
            b = rng.uniform(*_T["brightness"])
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*(b * np.array(_T["colour"])))])

            xf = UsdGeom.XformCommonAPI(mesh)
            xf.SetTranslate(Gf.Vec3d(*pos))
            xf.SetRotate(Gf.Vec3f(0, 0, rng.uniform(0, 360)))

            prim = mesh.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateAngularDampingAttr(ANGULAR_DAMPING)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
        if PILE_STYLE in ("mound", "crater", "pit"):
            build, name = {"mound": (_build_mound, "Mound"), "crater": (_build_crater, "Crater"),
                           "pit": (_build_pit, "Pit")}[PILE_STYLE]
            build(stage, f"{ROOT}/{name}", SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{ROOT}/{name}")).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
        elif PILE_STYLE == "box":
            _build_box(stage, ROOT + "/DigBox", SWING_AXIS_XY + PILE_DISTANCE * BOOM_DIR_XY,
                       BOOM_DIR_XY, DIG_BOX_SIZE, DIG_BOX_WALL_HEIGHT, (0.45, 0.33, 0.2))
        if DUMP_BIN:
            bin_dir = _rotate_xy(BOOM_DIR_XY, BIN_SWING_DEG)
            _build_box(stage, ROOT + "/Bin", SWING_AXIS_XY + BIN_DISTANCE * bin_dir,
                       bin_dir, BIN_SIZE, BIN_WALL_HEIGHT, (0.25, 0.35, 0.55))
        self.view = None
        self.initial_positions = positions
        self.initial_orientations = None
        print(f"[rocks] {NUM_ROCKS} {ROCK_TYPE} rocks, ~{ROCK_SIZE * 100:.1f} cm each"
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
