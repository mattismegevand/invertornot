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
    out = [json.loads(s) for s in r.lrange("bad", 0, -1)]
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
