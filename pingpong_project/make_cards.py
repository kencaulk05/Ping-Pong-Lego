"""
make_cards.py -- print the game's AprilTag cards (same tag family as HW2).

    python make_cards.py              # writes cards.pdf (4 pages) and card_0.png ... card_3.png

    tag 0  START
    tag 1  LEVEL 1  easy
    tag 2  LEVEL 2  medium
    tag 3  LEVEL 3  hard

Each card is one letter-size page with a 15 cm tag, big enough for the webcam to
read from a couple of meters away. Print at 100% ("actual size", not "fit to
page"), and keep the white border around the tag: the detector needs it.
"""

import argparse

import cv2
import numpy as np

from tag_input import LEVEL_TAGS, START_TAG, TAG_DICT
import config as C

DPI = 300
PAGE = (int(8.5 * DPI), int(11 * DPI))       # letter, portrait (width, height)


def label_for(tag_id):
    if tag_id == START_TAG:
        return "START", "show while idle: 3-2-1, then serve"
    lvl = LEVEL_TAGS[tag_id]
    level = C.LEVELS[lvl]
    return f"LEVEL {lvl}  {level.name.upper()}", f"ball arrives in {level.incoming_time:.1f} s"


def card(tag_id, size_cm=15.0):
    page = np.full((PAGE[1], PAGE[0]), 255, np.uint8)
    side = int(size_cm / 2.54 * DPI)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), tag_id, side)
    x0 = (PAGE[0] - side) // 2
    y0 = int(1.6 * DPI)
    page[y0:y0 + side, x0:x0 + side] = tag
    title, sub = label_for(tag_id)
    for text, y, scale, thick in ((title, int(1.1 * DPI), 4.0, 10), (sub, y0 + side + int(0.7 * DPI), 1.8, 4),
                                  (f"AprilTag 36h11, id {tag_id} - print at 100%", PAGE[1] - int(0.6 * DPI), 1.2, 2)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(page, text, ((PAGE[0] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)
    return page


def main():
    ap = argparse.ArgumentParser(description="Print the START and LEVEL cards")
    ap.add_argument("--size-cm", type=float, default=15.0)
    ap.add_argument("--out", default="cards.pdf")
    args = ap.parse_args()
    ids = [START_TAG] + sorted(LEVEL_TAGS)
    pages = [card(i, args.size_cm) for i in ids]
    for i, pg in zip(ids, pages):
        cv2.imwrite(f"card_{i}.png", pg)
    try:
        from PIL import Image
        imgs = [Image.fromarray(p).convert("RGB") for p in pages]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:], resolution=DPI)
        print(f"Saved {args.out} ({len(pages)} pages) and card_0.png ... card_{ids[-1]}.png")
    except ImportError:
        print(f"Saved card_0.png ... card_{ids[-1]}.png (install pillow for a single PDF)")
    print("Print at 100% scale (not 'fit to page').")


if __name__ == "__main__":
    main()
