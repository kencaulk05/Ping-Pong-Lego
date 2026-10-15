"""
mqtt_check.py -- prove the score really reaches the broker, before game day.

    python mqtt_check.py --name YourName

Connects to test.mosquitto.org, subscribes to your score topic and to a private
check topic, publishes a test message to the check topic, and times the round
trip. Also shows the score currently retained on ME193/Rogers/<name>, if any,
which is what a dashboard that connects now would see first.

    python mqtt_check.py --name YourName --publish 0.0     # also post a score by hand

Your score topic is only written with --publish, so the check never puts a fake
score in front of your instructor.
"""

import argparse
import threading
import time

from score_publisher import DEFAULT_BROKER, TOPIC_ROOT


def main():
    ap = argparse.ArgumentParser(description="Check the MQTT round trip")
    ap.add_argument("--name", required=True)
    ap.add_argument("--broker", default=DEFAULT_BROKER)
    ap.add_argument("--publish", type=float, default=None, help="also publish this score to your topic")
    args = ap.parse_args()

    from mqttlib import MQTTClient
    score_topic = f"{TOPIC_ROOT}/{args.name}"
    check_topic = f"{score_topic}/check"
    got = threading.Event()
    seen = {}

    def on_check(topic, payload):
        seen["rtt"] = time.perf_counter() - float(payload.split()[-1])
        got.set()

    def on_score(topic, payload):
        seen.setdefault("score", payload)

    client = MQTTClient(broker=args.broker)
    print(f"Connecting to {args.broker} ...")
    try:
        client.connect()
    except Exception as exc:
        raise SystemExit(f"FAILED: could not reach {args.broker}: {exc}")
    if not client.is_connected():
        raise SystemExit(f"FAILED: {args.broker} accepted the socket but never confirmed the MQTT connection.")
    print("Connected.")
    client.subscribe(check_topic, on_check)
    client.subscribe(score_topic, on_score)
    time.sleep(1.0)
    client.publish(check_topic, f"check {time.perf_counter()}")
    ok = got.wait(5.0)
    if ok:
        print(f"OK: a message went out and came back in {seen['rtt'] * 1000:.0f} ms.")
    else:
        print("FAILED: published, but nothing came back within 5 s.")
    print(f"Retained score on {score_topic}: {seen.get('score', '(none)')}")
    if args.publish is not None:
        client.publish(score_topic, f"{args.publish:.1f}", retain=True)
        time.sleep(0.5)
        print(f"Published {args.publish:.1f} to {score_topic} (retained).")
    client.disconnect()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
