"""Record videos from several fixed cameras while run_live.py runs.

Used by live_controller.py (V starts/stops). Each camera is a USD camera under
/World/RecCams (look through them from the viewport's camera menu). On the first
recording each gets an Isaac Camera sensor (render product + RGB annotator);
every `every` physics steps its frame goes to an MP4 (OpenCV, mp4v), plus one
grid video with all cameras side by side. Files: docs/videos/<time>_<camera>.mp4.
If an ffmpeg with libx264 is found they are re-encoded to H.264 afterwards
(plays everywhere, including browsers), in the background.

Videos follow sim time, not wall time: with physics at 60 Hz and every=2 they
play at 30 fps as if in real time, even when the sim itself runs slower.
The sensors are paused between recordings, but their render products stay
until run_live.py restarts, so the sim stays a little slower after the first V.
live_controller.py re-loads this file when it changes (between recordings).
"""
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "videos")
ROOT = "/World/RecCams"
FFMPEG_CANDIDATES = ["ffmpeg", os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg")]


def _ffmpeg():
    for c in FFMPEG_CANDIDATES:
        path = shutil.which(c) or (c if os.path.isfile(c) else None)
        if path:
            return path
    return None


class Recorder:
    def __init__(self, cameras, resolution=(640, 480), every=2, fps=30, grid=True):
        """cameras: {name: dict(eye=(x, y, z), target=(x, y, z), focal=mm)}."""
        self.cameras, self.resolution, self.every, self.fps, self.grid = cameras, tuple(resolution), every, fps, grid
        self.active = False
        self.sensors = {}
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Scope.Define(stage, ROOT)
        for name, c in cameras.items():
            cam = UsdGeom.Camera.Define(stage, f"{ROOT}/{name}")
            cam.CreateFocalLengthAttr(float(c.get("focal", 18.0)))
            cam.CreateHorizontalApertureAttr(20.955)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            self._aim(name)

    def _aim(self, name):
        c = self.cameras[name]
        set_camera_view(eye=np.array(c["eye"], float), target=np.array(c["target"], float),
                        camera_prim_path=f"{ROOT}/{name}")

    def start(self):
        if not self.sensors:
            from isaacsim.sensors.camera import Camera
            for name in self.cameras:
                cam = Camera(prim_path=f"{ROOT}/{name}", name=f"rec_{name}", resolution=self.resolution)
                cam.initialize()
                self._aim(name)  # initialize() resets the pose
                self.sensors[name] = cam
        else:
            for cam in self.sensors.values():
                cam.resume()
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        w, h = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writers, self.paths = {}, {}
        for name in self.cameras:
            self.paths[name] = os.path.abspath(os.path.join(OUT_DIR, f"{stamp}_{name}.mp4"))
            self.writers[name] = cv2.VideoWriter(self.paths[name], fourcc, self.fps, (w, h))
        if self.grid and len(self.cameras) > 1:
            self.cols = int(np.ceil(np.sqrt(len(self.cameras))))
            self.rows = int(np.ceil(len(self.cameras) / self.cols))
            self.paths["grid"] = os.path.abspath(os.path.join(OUT_DIR, f"{stamp}_grid.mp4"))
            self.writers["grid"] = cv2.VideoWriter(self.paths["grid"], fourcc, self.fps,
                                                   (w * self.cols, h * self.rows))
        self.frames, self.empty, self.count, self.last_bad = 0, 0, 0, None
        self.active = True
        print(f"[rec] recording {len(self.cameras)} cameras at {w}x{h}, {self.fps} fps -> {os.path.abspath(OUT_DIR)}")

    def step(self):
        if not self.active:
            return
        self.count += 1
        if self.count % self.every:
            return
        w, h = self.resolution
        tiles = []
        for name, cam in self.sensors.items():
            try:
                img = cam.get_rgba()
                img = np.zeros(0) if img is None else np.asarray(img)
            except Exception as e:
                img, self.last_bad = np.zeros(0), f"{name}: get_rgba() raised {e}"
            if img.size == 0 or img.ndim != 3:  # the first frames after starting are empty
                self.empty += 1
                if not self.last_bad or "raised" not in self.last_bad:
                    self.last_bad = f"{name}: got shape {img.shape}"
                if self.empty == 60 and self.frames == 0:
                    print(f"[rec] no images from the cameras yet ({self.last_bad})")
                return
            bgr = cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h))
            self.writers[name].write(bgr)
            tiles.append(cv2.putText(bgr.copy(), name, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2))
        if "grid" in self.writers:
            tiles += [np.zeros((h, w, 3), np.uint8)] * (self.rows * self.cols - len(tiles))
            rows = [np.hstack(tiles[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]
            self.writers["grid"].write(np.vstack(rows))
        if self.frames == 0:
            print(f"[rec] first frame captured (mean brightness {np.mean(tiles[0]):.0f}/255)")
        self.frames += 1

    def close(self):
        """Pause the sensors (their render products live until the app restarts)."""
        for cam in self.sensors.values():
            try:
                cam.pause()
            except Exception:
                pass

    def stop(self):
        if not self.active:
            return
        self.active = False
        for wr in self.writers.values():
            wr.release()
        self.close()
        print(f"[rec] stopped: {self.frames} frames ({self.frames / self.fps:.1f} s of video)")
        if not self.frames:
            print(f"[rec] nothing recorded: the cameras never returned an image ({self.empty} empty reads; "
                  f"last: {self.last_bad})")
            for p in self.paths.values():
                if os.path.exists(p):
                    os.remove(p)
            return
        for name, p in self.paths.items():
            print(f"[rec]   {name}: {p}")
        ff = _ffmpeg()
        if ff:
            # Re-encode to H.264 in the background (mp4v doesn't play in browsers).
            for p in self.paths.values():
                tmp = p[:-4] + "_h264.mp4"
                cmd = f'"{ff}" -loglevel error -y -i "{p}" -c:v libx264 -pix_fmt yuv420p "{tmp}" && mv "{tmp}" "{p}"'
                subprocess.Popen(cmd, shell=True)
            print("[rec] converting to H.264 in the background (a few seconds)")
