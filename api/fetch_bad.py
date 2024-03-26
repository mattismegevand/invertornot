#!/usr/bin/env python

import redis
import json
import os

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "decode_responses": True,
}
OUT_FILE = "bad.json"


if __name__ == "__main__":
    r = redis.Redis(**REDIS_CONFIG)
    reports_json = r.lrange("reports", 0, -1)
    new = [json.loads(report)["url"] for report in reports_json]
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r") as f:
            old = json.load(f)
        new = list(set(new + old))
    with open(OUT_FILE, "w") as f:
        json.dump(new, f, indent=2)
