from __future__ import annotations

import asyncio
import hashlib
import io
from typing import Any

import aiohttp
import redis
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import FileResponse

from nn import NN

REDIS_CONFIG = {"host": "localhost", "port": 6379, "decode_responses": True}
ONNX_MODEL_PATH = "model.onnx"

r = redis.Redis(**REDIS_CONFIG)
nn = NN(ONNX_MODEL_PATH)


def api_result(
    *, error: str = "", invert: int = -1, sha1: str = "", url: str = "",
) -> dict[Any, Any]:
    result = {"invert": invert, "sha1": sha1, "error": error}
    if url:
        result["url"] = url

    return result


def error(message: str, sha1: str = "", url: str = "") -> dict[Any, Any]:
    return api_result(
        error=message,
        sha1=sha1,
        url=url,
    )


async def fetch_image(url: str) -> dict[Any, Any] | tuple[bytes, str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, max_redirects=10) as response:
                # Check for non-success status
                if response.status != 200:
                    return error("Failed to fetch image from URL")

                content = await response.read()
                content_type = response.headers.get(
                    "Content-Type", "application/octet-stream",
                )

                return content, content_type
    except Exception:
        return error("Error: An unexpected error occurred while fetching the URL")


def process_content(
    content: bytes, content_type: str, *, url: str = "",
) -> dict[Any, Any]:
    if not content:
        return error("No image provided")

    if content_type not in ("image/jpg", "image/jpeg", "image/png"):
        return error("Only jpg, jpeg, and png (non-transparent) images are supported")

    sha1 = hashlib.sha1(content).hexdigest()

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
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def read_index():
        return FileResponse("index.html")

    @app.post("/api/file")
    async def process_files(files: list[UploadFile]) -> list[dict[Any, Any]]:
        results = []

        for file in files:
            content = await file.read()
            results.append(process_content(content, file.content_type))

        return results

    @app.post("/api/url")
    async def process_urls(urls: list[str]) -> list[dict[Any, Any]]:
        fetched = await asyncio.gather(*[fetch_image(url) for url in urls])

        results = []

        for url, res in zip(urls, fetched):
            if isinstance(res, dict):
                results.append(res)
                continue

            content, content_type = res
            results.append(process_content(content, content_type, url=url))

        return results

    @app.post("/api/sha1")
    async def process_sha1s(sha1s: list[str]) -> list[dict]:
        results = []

        for sha1 in sha1s:
            if (invert := r.get(sha1)) is None:
                results.append(error("No image found with the provided SHA1 hash"))
            else:
                results.append(api_result(invert=int(invert), sha1=sha1))

        return results

    return app


app = create_app()
