"""
score_publisher.py -- posts the record streak to MQTT.

Topic:    ME193/Rogers/<your name>
Payload:  the best streak as a float string, e.g. "7.0"
Broker:   test.mosquitto.org (what the class uses)

Fail-soft on purpose: if paho/mqttlib is missing or the broker is down, you get
a loud warning and the game keeps running, rather than crashing mid-rally.

It also doesn't take "connected" on faith. mqttlib's connect() waits up to 5 s
for the broker to confirm and then returns either way, so the publisher checks
is_connected(). If the broker hasn't confirmed, the score is held and sent as
soon as it does (paho keeps retrying in the background); `status` says which,
and the game shows it on screen.
"""

DEFAULT_BROKER = "test.mosquitto.org"
TOPIC_ROOT = "ME193/Rogers"


class ScorePublisher:
    def __init__(self, name, broker=DEFAULT_BROKER, enabled=True, retain=False, client=None):
        self.topic = f"{TOPIC_ROOT}/{name}"
        self.broker = broker
        self.retain = retain
        self.enabled = enabled
        self._client = client          # inject a fake in tests
        self._owns_client = False
        self.last_payload = None
        self._pending = None           # a score waiting for the broker to come back
        self.sent_count = 0

    def connect(self):
        if not self.enabled or self._client is not None:
            return
        try:
            from mqttlib import MQTTClient          # your course helper
            client = MQTTClient(broker=self.broker)
            client.connect()
            self._client = client
            self._owns_client = True
            if self._is_connected():
                print(f"[mqtt] connected to {self.broker}; publishing to {self.topic}")
            else:
                print(f"[mqtt] WARNING: {self.broker} hasn't confirmed the connection yet. "
                      "Scores will be held and sent when it does.")
        except Exception as exc:                     # missing paho, DNS, refused, ...
            print(f"[mqtt] WARNING: could not connect to {self.broker}: {exc}")
            print("[mqtt] The game will run, but no score will be published.")
            self.enabled = False

    def _is_connected(self):
        c = self._client
        if c is None:
            return False
        check = getattr(c, "is_connected", None)
        return True if check is None else bool(check())

    @property
    def status(self):
        if not self.enabled or self._client is None:
            return "off"
        if not self._is_connected():
            return "waiting for broker"
        return "ok"

    def publish_best(self, best):
        payload = f"{float(best):.1f}"
        self.last_payload = payload
        print(f"[mqtt] {self.topic} <- {payload}")
        if not self.enabled or self._client is None:
            return
        if not self._is_connected():
            self._pending = payload
            return
        self._send(payload)

    def tick(self):
        """Call every frame: sends a held score once the broker is back."""
        if self._pending is not None and self.enabled and self._is_connected():
            payload, self._pending = self._pending, None
            print(f"[mqtt] broker is back; sending {payload}")
            self._send(payload)

    def _send(self, payload):
        try:
            self._client.publish(self.topic, payload, retain=self.retain)
            self.sent_count += 1
        except Exception as exc:
            self._pending = payload
            print(f"[mqtt] WARNING: publish failed ({exc}); will retry")

    def close(self):
        if self._client is not None and self._owns_client:
            try:
                self._client.disconnect()
            except Exception:
                pass
