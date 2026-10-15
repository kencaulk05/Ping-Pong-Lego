"""
config.py -- every tunable number for the ping-pong game, in one place.

Coordinates are NORMALIZED screen coordinates: x and y both run 0..1 across
the window (or, later, across the webcam frame). That's what lets the same
game core run on a simulated mouse today and on MediaPipe wrist coordinates
tomorrow without changing a line of game logic.
"""

from dataclasses import dataclass

# --- Court geometry ---------------------------------------------------------
# Five landing zones across the near edge of the court. The opponent aims the
# ball at one of these; a dashed ring is drawn there for you.
ZONE_X = [0.10, 0.30, 0.50, 0.70, 0.90]
RING_Y = 0.62            # vertical position of the landing ring on screen (0 = top, 1 = bottom).
                         # Was 0.74; raised so you meet the ball farther up the court and can see the
                         # ring and the ball coming with room below it. The live game's --ring-y overrides it.
                         # Your hand doesn't move: the spot you calibrate with C always lines up with the ring.
RING_RADIUS = 0.12       # how close your wrist must be to the ring center, in units of
                         # screen HEIGHT (so the ring is a true circle on screen)
ASPECT = 960 / 600       # screen width / height, used to keep that circle circular

# --- Timing -----------------------------------------------------------------
HIT_WINDOW = 0.20        # swing counts if it lands within +/- this many seconds of the
                         # ball's arrival. Widen it if BLE/IMU latency makes timing mushy.
SWING_REPORT_DELAY = 0.0 # extra seconds to wait past the hit window before calling "no swing". A keyboard
                         # swing is instant, so 0. The IMU recognizer only reports a swing ~0.26 s after
                         # its peak (measured 210-272 ms on your hub), so pingpong_live.py uses LIVE_SWING_REPORT_DELAY.
LIVE_SWING_REPORT_DELAY = 0.30
HAPTIC_LEAD = 0.10       # the hub turns green this long before the swing window opens, to cover the
                         # Bluetooth trip, so the light is green by the time the window really opens
SWING_COOLDOWN = 0.40    # after any swing, further swings are ignored this long (stops
                         # spam-swinging, and means a too-early swing really wastes your shot)
COUNTDOWN_SECONDS = 3.0
RESERVE_AFTER_MISS = 1.6     # pause before the next serve after you miss
RESERVE_AFTER_WINNER = 1.2   # pause before the next serve after you win the point
POSITION_LOOKBACK = 0.30    # you count as "in the ring" if your wrist was inside it at any moment in the
                             # last 0.3 s before the swing's peak. A swing carries the hand away from where
                             # you lined it up (your first live game lost two well-timed backhands that way).
POSE_LOST_TIMEOUT = 5.0      # ball freezes while your wrist isn't detected; after this many
                             # seconds the game gives up and goes back to idle

# --- Your shot --------------------------------------------------------------
# Swing type decides where your shot goes, relative to where you hit it from.
# (Screen-right is positive. Flip the signs if forehand feels backwards.)
SHOT_OFFSET = {"forehand": +0.25, "backhand": -0.25, "flat": 0.0}
# For a right-handed player facing the screen, a forehand sweeps the paddle right-to-left
# across the body, so the ball goes to the left; a backhand sends it right. The live game
# uses this for --hand right (and mirrors it for --hand left); --flip-shots reverses it.
SHOT_OFFSET_RIGHT_HANDED = {"forehand": -0.25, "backhand": +0.25, "flat": 0.0}
AIM_JITTER = 0.10        # std-dev of aiming error when you're at the very edge of the ring;
                         # shrinks to zero as your wrist gets closer to the ring center
RET_TIME_SLOW = 1.5      # seconds for your shot to reach the far wall at power 0
RET_TIME_FAST = 0.6      # ...and at power 1 (harder hit = less time for the opponent)

# --- Opponent's shot --------------------------------------------------------
FAST_SHOT_TIME_MULT = 0.7    # a "fast" opponent shot arrives in 70% of the normal time

# --- Reward the opponent policy gets for its shot ---------------------------
# (Used by the learning opponent in stage 3; the rule-based one ignores it.)
REWARD_PLAYER_MISSED = +1.0
REWARD_PLAYER_RETURNED = -0.2


@dataclass(frozen=True)
class Level:
    name: str
    incoming_time: float     # seconds for the opponent's shot to reach you
    opp_speed: float         # how fast the opponent slides, in screen-widths per second
    opp_reaction: float      # seconds before the opponent starts moving after you hit
    opp_half_width: float    # half the opponent's paddle width (its reach), screen-widths
    aggression: float        # chance the rule-based opponent aims away from you
    fast_prob: float         # chance the rule-based opponent hits a fast shot
    q_fast: bool = True      # may the learning opponent hit fast shots at this level?
    hit_window: float = HIT_WINDOW   # +/- seconds a swing may be off the ball's arrival
    ring_scale: float = 1.0          # how many times RING_RADIUS your wrist may be from the ring center


# What the AprilTag "level card" selects. Tag 1 / 2 / 3.
LEVELS = {
    1: Level("easy",   incoming_time=2.4, opp_speed=0.60, opp_reaction=0.30,
             opp_half_width=0.10, aggression=0.30, fast_prob=0.10, q_fast=False, hit_window=0.30, ring_scale=1.5),
    2: Level("medium", incoming_time=1.7, opp_speed=0.65, opp_reaction=0.18,
             opp_half_width=0.11, aggression=0.55, fast_prob=0.30, hit_window=0.20, ring_scale=1.35),
    3: Level("hard",   incoming_time=1.2, opp_speed=0.90, opp_reaction=0.10,
             opp_half_width=0.12, aggression=0.80, fast_prob=0.50, hit_window=0.15, ring_scale=1.2),
}
# ring_scale: in 7 live games, 87 of 186 misses were "out of position", and 72% of those were within 1.5 ring
# radii of the edge (a near miss, not a wrong place); 63 of the 87 were at level 3. Scaling the ring by level
# (easy 1.5, medium 1.35, hard 1.2) keeps the harder levels stricter and would have turned 39 of those misses
# into hits (hit rate 62% -> 70%). A single x1.5 for every level would reach 74% but flatten the difficulty
# ladder. The ring is drawn at the scaled size, so what you see is what counts.
