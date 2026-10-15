"""
pose_input.py -- where is your paddle hand? (MediaPipe Pose, from HW1)

Two pieces:

  PoseTracker   runs MediaPipe's PoseLandmarker (the Tasks API your HW1 arm_race.py
                uses, model pose_landmarker_lite.task) on each webcam frame and returns
                your paddle wrist in MIRRORED camera coordinates (0..1), so moving your
                hand to your right moves it right on screen, like a mirror.

  WristMapper   turns that camera position into a position on the virtual court. You
                don't have to stand so your hand happens to land where the game draws
                the ring: press C while holding the hub where a comfortable hit is, and
                that spot becomes the middle of the hitting line. `reach` is how much of
                the camera's width, side to side, spans the whole court.

Landmark labels: MediaPipe calls a landmark "right wrist" for the person's own right
wrist, but only on an UNmirrored image. So the model runs on the raw frame and the x
coordinate is mirrored afterwards; flipping the frame first would swap the hands (HW1
had to work around exactly that). --swap-hands is there in case your camera driver
already mirrors the image.
"""

import os
import time
import urllib.request

# MediaPipe and TensorFlow Lite print harmless start-up warnings ("Feedback manager requires a
# model with a single signature...", "Using NORM_RECT without IMAGE_DIMENSIONS..."). Hide them.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config as C

POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

LEFT_WRIST, RIGHT_WRIST = 15, 16
# For the little skeleton drawn in the camera preview.
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]

MIN_VISIBILITY = 0.5     # a wrist MediaPipe is less sure of than this counts as "not seen"
HOLD_S = 0.25            # keep the last wrist this long through a dropout before reporting it lost
SMOOTH = 0.6             # 1 = raw (most responsive), lower = smoother but laggier


def ensure_pose_model(path=POSE_MODEL_PATH):
    if not os.path.exists(path):
        print("Downloading the MediaPipe pose model (one time)...")
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path


class PoseTracker:
    def __init__(self, hand="right", swap_hands=False, model_path=POSE_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        self._mp = mp
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_pose_model(model_path)),
            running_mode=RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = PoseLandmarker.create_from_options(options)
        right = (hand == "right") != swap_hands
        self.idx = RIGHT_WRIST if right else LEFT_WRIST
        self._t0 = time.monotonic()
        self._last_ms = -1
        self.filter = WristFilter()
        self.landmarks = None          # mirrored (x, y, visibility) for every landmark, for drawing

    def process(self, frame_bgr, now=None):
        """Returns the paddle wrist as mirrored, normalized (x, y), or None."""
        import cv2
        now = time.monotonic() if now is None else now
        ms = int((now - self._t0) * 1000)
        ms = max(ms, self._last_ms + 1)          # VIDEO mode needs strictly increasing timestamps
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, ms)
        raw = None
        self.landmarks = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            self.landmarks = [(1.0 - p.x, p.y, getattr(p, "visibility", 1.0)) for p in lm]
            x, y, vis = self.landmarks[self.idx]
            if vis >= MIN_VISIBILITY and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                raw = (x, y)
        return self.filter.update(raw, now)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


class WristFilter:
    """Light smoothing, and riding out one- or two-frame dropouts. A tracking blink
    shouldn't freeze the ball, but a hand that is really gone should."""

    def __init__(self, smooth=SMOOTH, hold_s=HOLD_S):
        self.smooth, self.hold_s = smooth, hold_s
        self.value = None
        self.last_seen = None

    def update(self, raw, now):
        if raw is None:
            if self.value is not None and self.last_seen is not None and now - self.last_seen <= self.hold_s:
                return self.value
            self.value = None
            return None
        if self.value is None:
            self.value = raw
        else:
            a = self.smooth
            self.value = (a * raw[0] + (1 - a) * self.value[0], a * raw[1] + (1 - a) * self.value[1])
        self.last_seen = now
        return self.value


class WristMapper:
    """Camera position (mirrored, 0..1) -> court position (0..1, the game's coordinates).

    The calibrated spot maps to the middle of the hitting line (x 0.5, y RING_Y).
    Moving `reach` of the camera's width covers the whole court side to side. Vertical
    movement uses the same physical scale, so the ring stays a circle in real space."""

    def __init__(self, cx=0.5, cy=0.55, reach=0.6, cam_aspect=4 / 3):
        self.cx, self.cy, self.reach, self.cam_aspect = cx, cy, reach, cam_aspect

    def calibrate(self, cam_xy):
        self.cx, self.cy = cam_xy

    def to_court(self, cam_xy):
        if cam_xy is None:
            return None
        x = 0.5 + (cam_xy[0] - self.cx) / self.reach
        # a camera-height unit is 1/cam_aspect of a camera-width unit; a court-height
        # unit is 1/ASPECT of a court-width unit
        y = C.RING_Y + (cam_xy[1] - self.cy) / self.cam_aspect * C.ASPECT / self.reach
        return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    def to_camera(self, court_xy):
        """The inverse, for drawing the court's hitting line on the camera preview."""
        x = self.cx + (court_xy[0] - 0.5) * self.reach
        y = self.cy + (court_xy[1] - C.RING_Y) * self.reach * self.cam_aspect / C.ASPECT
        return (x, y)

    def as_dict(self):
        return {"cx": self.cx, "cy": self.cy, "reach": self.reach}
