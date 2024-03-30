#!/usr/bin/env python

import asyncio
import hashlib
import io
import json
import os
from typing import Any

import aiohttp
import redis
from PIL import Image

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "decode_responses": True,
}
OUT_FILE = "bad.json"


async def save_image_as_png(content, path):
    image = Image.open(io.BytesIO(content))
    if not path.endswith(".png"):
        path = os.path.splitext(path)[0] + ".png"
    image.save(path, "PNG")


async def main():
    r = redis.Redis(**REDIS_CONFIG)
    out = [json.loads(s) for s in r.lrange("bad", 0, -1)]
    os.makedirs("corrections", exist_ok=True)
    os.makedirs("corrections/brightness", exist_ok=True)
    os.makedirs("corrections/invert", exist_ok=True)
    for url, correct in [o.values() for o in out]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, max_redirects=10) as response:
                if response.status != 200:
                    print(f"Error: HTTP status {response.status} for URL {url}")
                content = await response.read()
                sha1 = hashlib.sha1(content).hexdigest()
                path = f"corrections/{'invert' if correct else 'brightness'}/{sha1}.png"
                if os.path.exists(path):
                    continue
                await save_image_as_png(content, path)


if __name__ == "__main__":
    asyncio.run(main())
