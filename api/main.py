from __future__ import annotations

import asyncio
import hashlib
import io
import json
from contextlib import asynccontextmanager
from os import getenv
from typing import Any

import aiohttp
import redis
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
from nn import NN
from PIL import Image
from starlette.responses import FileResponse

REDIS_CONFIG = {
    "host": getenv("REDIS_HOST", "localhost"),
    "port": int(getenv("REDIS_PORT", 6379)),
    "decode_responses": True,
}
ONNX_MODEL_PATH = getenv("ONNX_MODEL_PATH", "model.onnx")
FETCH_MAX_SIZE = int(getenv("FETCH_MAX_SIZE", 25 * 1024 * 1024))
MAX_IMAGES_PER_REQUEST = int(getenv("MAX_IMAGES_PER_REQUEST", 250))

r: redis.Redis = None
nn: NN = None


def api_result(
    *,
    error: str = "",
    invert: int = -1,
    sha1: str = "",
    url: str = "",
) -> dict[str, Any]:
    result = {"invert": invert, "sha1": sha1, "error": error}
    if url:
        result["url"] = url

    return result


def error(message: str, sha1: str = "", url: str = "") -> dict[str, Any]:
    return api_result(
        error=message,
        sha1=sha1,
        url=url,
    )


async def fetch_image(url: str) -> dict[str, Any] | tuple[bytes, str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, max_redirects=10) as response:
                # Check for non-success status
                if response.status != 200:
                    return error("Failed to fetch image from URL")

                content_length = response.headers.get("Content-Length")
                if content_length:
                    content_length = int(content_length)
                    if content_length > FETCH_MAX_SIZE:
                        return {"error": "File size exceeds limit"}

                content = await response.read()
                content_type = response.headers.get(
                    "Content-Type",
                    "application/octet-stream",
                )

                return content, content_type
    except Exception:
        return error("Error: An unexpected error occurred while fetching the URL")


def process_content(
    content: bytes,
    content_type: str,
    *,
    url: str = "",
) -> dict[str, Any]:
    if not content:
        return error("No image provided")

    if content_type not in ("image/jpg", "image/jpeg", "image/png"):
        return error("Only jpg, jpeg, and png (non-transparent) images are supported")

    sha1 = hashlib.sha1(content).hexdigest()
    if url:
        r.set(url, sha1)

    if (invert := r.get(sha1)) is not None:
        return api_result(invert=int(invert), sha1=sha1, url=url)

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except OSError:
        return error("Invalid image format", sha1=sha1, url=url)
    except Exception:
        return error(
            "An unexpected error occurred while processing the image",
            sha1=sha1,
            url=url,
        )

    invert = nn.pred(image)
    r.set(sha1, invert)

    return api_result(invert=invert, sha1=sha1, url=url)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global r, nn
        r = redis.Redis(**REDIS_CONFIG)
        nn = NN(ONNX_MODEL_PATH)
        yield

    app = FastAPI(title="invertornot.com", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

    @app.get("/", include_in_schema=False)
    async def read_index():
        return FileResponse("static/index.html")

    @app.post("/api/file")
    async def process_files(files: list[UploadFile]) -> list[dict[str, Any]]:
        if len(files) > MAX_IMAGES_PER_REQUEST:
            return [error("Too many files provided")]
        results = []

        for file in files:
            content = await file.read()
            results.append(process_content(content, file.content_type))

        return results

    @app.post("/api/url", response_class=ORJSONResponse)
    async def process_urls(urls: list[str]) -> list[dict[str, Any]]:
        urls_json = json.dumps(urls)
        if (cached := r.get(urls_json)) is not None:
            return ORJSONResponse(json.loads(cached))

        if len(urls) > MAX_IMAGES_PER_REQUEST:
            return [error("Too many URLs provided")]

        results = [
            api_result(invert=int(invert), sha1=sha1, url=url)
            for url in urls
            if (sha1 := r.get(url)) is not None and (invert := r.get(sha1)) is not None
        ]
        urls = [url for url in urls if url not in [result["url"] for result in results]]
        fetched = await asyncio.gather(*[fetch_image(url) for url in urls])

        for url, res in zip(urls, fetched):
            if isinstance(res, dict):
                results.append(res)
                continue

            content, content_type = res
            results.append(process_content(content, content_type, url=url))

        r.set(urls_json, json.dumps(results), ex=86400)

        return results

    @app.post("/api/sha1")
    async def process_sha1s(sha1s: list[str]) -> list[dict]:
        results = []

        for sha1 in sha1s:
            if (invert := r.get(sha1)) is not None:
                results.append(api_result(invert=int(invert), sha1=sha1))
            else:
                results.append(error("No image found with the provided SHA1 hash"))

        return results

    @app.post("/api/correction")
    async def correction(urls: list[str]) -> None:
        for url in urls:
            if (sha1 := r.get(url)) is not None and (invert := r.get(sha1)) is not None:
                r.lpush("correction", json.dumps({"url": url, "correct": 1 - int(invert)}))
        return

    return app


app = create_app()
