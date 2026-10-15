"""
tag_input.py -- the level and start cards (AprilTags, from HW2).

Same detector as HW2's apriltag_center.py: OpenCV's ArUco module with the
DICT_APRILTAG_36h11 family. Print the cards with `python make_cards.py`.

    tag 0  START   (shown while idle: 3-2-1 countdown, then the first serve)
    tag 1  LEVEL 1 easy
    tag 2  LEVEL 2 medium
    tag 3  LEVEL 3 hard

A tag has to be seen in STABLE_FRAMES frames in a row before it counts, so one
misread frame can't start a game or change the level. The start card fires once
per showing: keep holding it up and nothing more happens; lower it and show it
again to start again.
"""

import cv2

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
START_TAG = 0
LEVEL_TAGS = {1: 1, 2: 2, 3: 3}          # tag id -> game level
STABLE_FRAMES = 3


class TagReader:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is missing from this OpenCV build: pip install opencv-contrib-python")
        dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.last = []                   # [(id, corners (4, 2) in raw pixels)] from the latest frame

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last = [] if ids is None else [(int(i), c[0]) for i, c in zip(ids.flatten(), corners)]
        return {i for i, _ in self.last}


class CardDebouncer:
    """Seen tag ids per frame -> (level_card, start_card) for the game."""

    def __init__(self, stable=STABLE_FRAMES):
        self.stable = stable
        self.streak = {}                 # tag id -> consecutive frames seen
        self.start_armed = True

    def update(self, ids):
        for i in list(self.streak):
            if i not in ids:
                del self.streak[i]
        for i in ids:
            self.streak[i] = self.streak.get(i, 0) + 1

        level = None
        for tag, lvl in LEVEL_TAGS.items():
            if self.streak.get(tag, 0) >= self.stable:
                level = lvl
        start = False
        if self.streak.get(START_TAG, 0) >= self.stable:
            if self.start_armed:
                start, self.start_armed = True, False
        elif START_TAG not in self.streak:
            self.start_armed = True
        return level, start
